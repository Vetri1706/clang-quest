"""A deterministic, byte-complete byte-pair encoder using the standard library.

The initial vocabulary contains three special symbols and all 256 bytes. BPE
merges never cross input-document boundaries. Training explicitly limits bytes
retained in memory; this is a local reference implementation, not a parallel BPE
trainer. Encoding repeatedly applies the lowest-ranked available adjacent merge.
"""

from __future__ import annotations

import base64
from collections import Counter
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable, Sequence
from .safeio import read_json


class ByteBPETokenizer:
    pad_id = 0
    bos_id = 1
    eos_id = 2
    byte_offset = 3
    base_vocab_size = 259

    def __init__(self, merges: Sequence[Sequence[int]] = (), *, max_token_bytes=4096,
                 max_vocab_bytes=16 * 1024**2, max_merges=65536):
        if any(type(n) is not int or n < 1 for n in (max_token_bytes, max_vocab_bytes, max_merges)) or len(merges) > max_merges:
            raise ValueError("BPE merge graph exceeds the configured resource bounds")
        self.merges: list[tuple[int, int]] = []
        self.vocab: dict[int, bytes] = {i + 3: bytes([i]) for i in range(256)}
        self.ranks: dict[tuple[int, int], tuple[int, int]] = {}
        vocabulary_bytes = 256
        for rank, pair in enumerate(merges):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2 or any(type(n) is not int for n in pair):
                raise ValueError("Every BPE merge must contain exactly two token IDs")
            left, right = pair
            if left not in self.vocab or right not in self.vocab:
                raise ValueError("A BPE merge may only reference existing byte tokens")
            if (left, right) in self.ranks:
                raise ValueError("Duplicate BPE merge")
            token_bytes = len(self.vocab[left]) + len(self.vocab[right])
            vocabulary_bytes += token_bytes
            if token_bytes > max_token_bytes or vocabulary_bytes > max_vocab_bytes:
                raise ValueError("Expanded BPE vocabulary exceeds its byte limit")
            token = self.base_vocab_size + rank
            self.merges.append((left, right))
            self.ranks[(left, right)] = (rank, token)
            self.vocab[token] = self.vocab[left] + self.vocab[right]
        self.training_bytes = 0

    @property
    def vocab_size(self) -> int:
        return self.base_vocab_size + len(self.merges)

    @staticmethod
    def _replace(tokens: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
        output: list[int] = []
        position = 0
        while position < len(tokens):
            if position + 1 < len(tokens) and (tokens[position], tokens[position + 1]) == pair:
                output.append(new_id)
                position += 2
            else:
                output.append(tokens[position])
                position += 1
        return tuple(output)

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int = 512, max_bytes: int = 2_000_000,
              min_frequency: int = 2) -> "ByteBPETokenizer":
        if type(vocab_size) is not int or not cls.base_vocab_size <= vocab_size <= cls.base_vocab_size + 65536:
            raise ValueError("vocab_size must accommodate the byte vocabulary and at most 65536 merges")
        if any(type(n) is not int or n < 1 for n in (max_bytes, min_frequency)):
            raise ValueError("max_bytes and min_frequency must be positive")
        corpus: Counter[tuple[int, ...]] = Counter()
        retained = 0
        for text in texts:
            if not isinstance(text, str):
                raise TypeError("BPE training expects decoded text strings")
            remaining = max_bytes - retained
            if remaining <= 0:
                break
            # Slice characters before UTF-8 encoding so one giant supplied string
            # cannot allocate an unbounded additional encoded copy.
            raw = text[:remaining].encode("utf-8")[:remaining]
            retained += len(raw)
            if raw:
                corpus[tuple(byte + cls.byte_offset for byte in raw)] += 1
        merges: list[tuple[int, int]] = []
        while cls.base_vocab_size + len(merges) < vocab_size:
            counts: Counter[tuple[int, int]] = Counter()
            for sequence, frequency in corpus.items():
                for left, right in zip(sequence, sequence[1:]):
                    counts[(left, right)] += frequency
            if not counts:
                break
            pair, frequency = min(counts.items(), key=lambda item: (-item[1], item[0]))
            if frequency < min_frequency:
                break
            new_id = cls.base_vocab_size + len(merges)
            merges.append(pair)
            replaced: Counter[tuple[int, ...]] = Counter()
            for sequence, count in corpus.items():
                replaced[cls._replace(sequence, pair, new_id)] += count
            corpus = replaced
        result = cls(merges)
        result.training_bytes = retained
        return result

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("encode expects a string")
        sequence = tuple(byte + self.byte_offset for byte in text.encode("utf-8"))
        while len(sequence) > 1:
            candidate: tuple[int, tuple[int, int], int] | None = None
            for pair in zip(sequence, sequence[1:]):
                merge = self.ranks.get(pair)
                if merge is not None and (candidate is None or merge[0] < candidate[0]):
                    candidate = (merge[0], pair, merge[1])
            if candidate is None:
                break
            sequence = self._replace(sequence, candidate[1], candidate[2])
        return ([self.bos_id] if add_bos else []) + list(sequence) + ([self.eos_id] if add_eos else [])

    def decode(self, tokens: Iterable[int], skip_special: bool = True,
               errors: str = "replace") -> str:
        special = {self.pad_id: b"<pad>", self.bos_id: b"<bos>", self.eos_id: b"<eos>"}
        parts: list[bytes] = []
        for item in tokens:
            token = int(item)
            if token in special:
                if not skip_special:
                    parts.append(special[token])
            elif token in self.vocab:
                parts.append(self.vocab[token])
            else:
                raise ValueError(f"Token ID {token} is outside the vocabulary")
        return b"".join(parts).decode("utf-8", errors=errors)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": "rawllm-byte-bpe", "version": 1,
                   "special_ids": {"pad": 0, "bos": 1, "eos": 2},
                   "merges": self.merges, "training_bytes": self.training_bytes,
                   "vocabulary": {str(key): base64.b64encode(value).decode("ascii")
                                  for key, value in sorted(self.vocab.items())}}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                             prefix=f".{path.name}.", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        payload = read_json(path, max_bytes=32 * 1024**2)
        if payload.get("format") != "rawllm-byte-bpe" or payload.get("version") != 1:
            raise ValueError("Unsupported tokenizer serialization format")
        if payload.get("special_ids") != {"pad": 0, "bos": 1, "eos": 2}:
            raise ValueError("Tokenizer special IDs do not match this implementation")
        result = cls(payload["merges"])
        expected = {str(key): base64.b64encode(value).decode("ascii")
                    for key, value in sorted(result.vocab.items())}
        if payload.get("vocabulary") != expected:
            raise ValueError("Tokenizer vocabulary does not match its merge graph")
        training_bytes = payload.get("training_bytes", 0)
        if type(training_bytes) is not int or training_bytes < 0:
            raise ValueError("Tokenizer training byte count must be a nonnegative integer")
        result.training_bytes = training_bytes
        return result
