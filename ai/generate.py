"""Bounded paged-cache generation from this framework's own checkpoint format."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(key, "1")
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from rawllm.model import Config, Transformer
from rawllm.cache import PagedLatentCache
from rawllm.tokenizer import ByteBPETokenizer
from rawllm.safeio import read_json, read_npz


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_run(run):
    run = Path(run)
    settings = read_json(run / "settings.json")
    if file_hash(run / "tokenizer.json") != settings["tokenizer_sha256"]:
        raise ValueError("Tokenizer checksum mismatch.")
    tokenizer = ByteBPETokenizer.load(run / "tokenizer.json")
    model = Transformer(Config(**settings["config"]))
    manifest = read_json(run / "checkpoint/manifest.json")
    if manifest.get("format") != "rawllm-checkpoint" or manifest.get("version") != 1:
        raise ValueError("Unsupported checkpoint format")
    filename = manifest["payload"]
    if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".npz"):
        raise ValueError("Invalid checkpoint payload path.")
    path = run / "checkpoint" / filename
    names = sorted(model.params)
    if manifest["parameters"] != names:
        raise ValueError("Checkpoint parameter names do not match the model.")
    if len(set(manifest["gradients"])) != len(manifest["gradients"]) or not set(manifest["gradients"]).issubset(names):
        raise ValueError("Checkpoint gradient inventory is invalid")
    expected, selected = {}, []
    for index, name in enumerate(names):
        shape = model.params[name].shape
        key = f"parameter_{index}"
        selected.append(key)
        expected[key] = (shape, (np.float32, np.float64))
        for group in ("master", "m", "v"):
            expected[f"{group}_{index}"] = (shape, np.float32)
        if name in manifest["gradients"]:
            expected[f"gradient_{index}"] = (shape, (np.float16, np.float32, np.float64))
    arrays = read_npz(path, expected, sha256=manifest["sha256"], select=selected)
    for index, name in enumerate(names):
        value = arrays[f"parameter_{index}"]
        if np.any(np.abs(value) > np.finfo(np.float32).max):
            raise ValueError("Checkpoint parameter exceeds FP32 inference storage")
        model.params[name].data[...] = value
    return model, tokenizer


def generate(model, tokenizer, prompt, max_tokens=32, temperature=.8, seed=5):
    if not 0 <= max_tokens <= 128 or not np.isfinite(temperature) or temperature < 0:
        raise ValueError("Use 0..128 tokens and finite nonnegative temperature.")
    ids = tokenizer.encode(prompt, add_bos=True)
    if len(ids) + max_tokens > model.config.max_seq_len:
        raise ValueError("Prompt plus generation exceeds model context; no silent truncation.")
    cache = PagedLatentCache(model.config, page_size=8,
                             max_pages=(len(ids) + max_tokens + 7) // 8)
    rng = np.random.default_rng(seed)
    for token in ids:
        logits = model.decode(token, cache)
    generated = []
    for _ in range(max_tokens):
        scores = logits.astype(np.float64)
        scores[tokenizer.pad_id] = -np.inf
        scores[tokenizer.bos_id] = -np.inf
        if temperature == 0:
            token = int(scores.argmax())
        else:
            scores = scores / temperature
            probabilities = np.exp(scores - scores.max())
            probabilities /= probabilities.sum()
            token = int(rng.choice(len(probabilities), p=probabilities))
        if token == tokenizer.eos_id:
            break
        generated.append(token)
        logits = model.decode(token, cache)
    cache.release("default")
    return tokenizer.decode(generated)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("prompt")
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=.8)
    a = p.parse_args()
    model, tokenizer = load_run(a.run)
    print(json.dumps({"generated_text": generate(model, tokenizer, a.prompt, a.tokens, a.temperature),
                      "status": "unvalidated_tiny_model_output"}, ensure_ascii=True))


if __name__ == "__main__":
    main()
