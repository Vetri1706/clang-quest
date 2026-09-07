"""Bounded streaming text ingestion and causal document/SFT packing.

Plain text is decoded incrementally in bounded chunks; each emitted chunk is a
document for packing. JSONL preserves each record as a document and rejects a
record longer than max_line_bytes. Neither reader loads a complete file. Packing
holds one bounded window plus the current supplied document's tokenization.
"""

from __future__ import annotations

import codecs
from collections import deque
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np

from .tokenizer import ByteBPETokenizer


@dataclass(frozen=True)
class PackedBatch:
    input_ids: np.ndarray
    targets: np.ndarray
    loss_mask: np.ndarray
    segment_ids: np.ndarray
    attention_mask: np.ndarray

    @property
    def supervised_tokens(self) -> int:
        return int(self.loss_mask.sum())


def iter_text(paths: str | Path | Sequence[str | Path], chunk_bytes: int = 65_536,
              jsonl_field: str = "text", max_line_bytes: int = 8_388_608) -> Iterator[str]:
    if chunk_bytes < 1 or max_line_bytes < 1:
        raise ValueError("Streaming byte limits must be positive")
    if isinstance(paths, (str, Path)):
        paths = [paths]
    for entry in paths:
        path = Path(entry)
        compressed = path.suffix.lower() == ".gz"
        actual_suffix = path.with_suffix("").suffix.lower() if compressed else path.suffix.lower()
        opener = gzip.open if compressed else open
        with opener(path, "rb") as handle:
            if actual_suffix == ".jsonl":
                number = 0
                while True:
                    line = handle.readline(max_line_bytes + 1)
                    if not line:
                        break
                    number += 1
                    if len(line) > max_line_bytes:
                        raise ValueError(f"{path}: JSONL record {number} exceeds {max_line_bytes} bytes")
                    if not line.strip():
                        continue
                    record = json.loads(line.decode("utf-8", errors="strict"))
                    if not isinstance(record, dict) or not isinstance(record.get(jsonl_field), str):
                        raise ValueError(f"{path}: JSONL record {number} lacks text field {jsonl_field!r}")
                    yield record[jsonl_field]
            else:
                decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
                while True:
                    raw = handle.read(chunk_bytes)
                    if not raw:
                        tail = decoder.decode(b"", final=True)
                        if tail:
                            yield tail
                        break
                    decoded = decoder.decode(raw, final=False)
                    if decoded:
                        yield decoded


def document_attention_mask(segment_ids: np.ndarray) -> np.ndarray:
    segments = np.asarray(segment_ids, dtype=np.int64)
    if segments.ndim != 2:
        raise ValueError("segment_ids must have shape [batch, sequence]")
    length = segments.shape[1]
    causal = np.arange(length)[None, :] <= np.arange(length)[:, None]
    equal = segments[:, :, None] == segments[:, None, :]
    real = (segments[:, :, None] >= 0) & (segments[:, None, :] >= 0)
    # Padding queries attend to themselves so a row never has an empty softmax.
    pad_diagonal = (segments[:, :, None] < 0) & np.eye(length, dtype=bool)[None, :, :]
    return ((equal & real & causal[None, :, :]) | pad_diagonal)[:, None, :, :]


def _packed(records: Iterable[tuple[list[int], list[float]]], tokenizer: ByteBPETokenizer,
            sequence_length: int, drop_last: bool) -> Iterator[PackedBatch]:
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    window: deque[tuple[int, float, int]] = deque()

    def render(items: list[tuple[int, float, int]]) -> PackedBatch:
        padded = items + [(tokenizer.pad_id, 0.0, -1)] * (sequence_length + 1 - len(items))
        ids = np.asarray([item[0] for item in padded], dtype=np.int64)
        supervision = np.asarray([item[1] for item in padded], dtype=np.float32)
        segments = np.asarray([item[2] for item in padded], dtype=np.int64)
        valid_transition = (segments[:-1] == segments[1:]) & (segments[1:] >= 0)
        mask = supervision[1:] * valid_transition
        input_segments = segments[:-1][None, :]
        return PackedBatch(ids[:-1][None, :], ids[1:][None, :], mask[None, :],
                           input_segments, document_attention_mask(input_segments))

    for segment, (tokens, supervision) in enumerate(records):
        if len(tokens) != len(supervision):
            raise ValueError("Each token requires a corresponding supervision flag")
        for token, flag in zip(tokens, supervision):
            window.append((int(token), float(flag), segment))
            if len(window) == sequence_length + 1:
                yield render(list(window))
                # Retain the last token as next window's context. Targets do not repeat.
                for _ in range(sequence_length):
                    window.popleft()
    if len(window) > 1 and not drop_last:
        yield render(list(window))


def pack_documents(documents: Iterable[str], tokenizer: ByteBPETokenizer,
                   sequence_length: int, drop_last: bool = False) -> Iterator[PackedBatch]:
    def records() -> Iterator[tuple[list[int], list[float]]]:
        for document in documents:
            tokens = tokenizer.encode(document, add_bos=True, add_eos=True)
            yield tokens, [0.0] + [1.0] * (len(tokens) - 1)
    return _packed(records(), tokenizer, sequence_length, drop_last)


def pack_sft(records: Iterable[Mapping[str, str]], tokenizer: ByteBPETokenizer,
             sequence_length: int, drop_last: bool = False) -> Iterator[PackedBatch]:
    def token_records() -> Iterator[tuple[list[int], list[float]]]:
        for record in records:
            prompt, response = record["prompt"], record["response"]
            if not isinstance(prompt, str) or not isinstance(response, str):
                raise TypeError("SFT prompt and response must be strings")
            # Encoding separately creates an explicit prompt/response boundary and
            # prevents a BPE merge from absorbing supervised response bytes.
            prompt_ids = tokenizer.encode(prompt, add_bos=True)
            response_ids = tokenizer.encode(response, add_eos=True)
            yield prompt_ids + response_ids, [0.0] * len(prompt_ids) + [1.0] * len(response_ids)
    return _packed(token_records(), tokenizer, sequence_length, drop_last)


def preference_batch(prompt: str, response: str, tokenizer: ByteBPETokenizer,
                     max_length: int) -> PackedBatch:
    """Construct one untruncated DPO completion; reject silent completion truncation."""
    required = len(tokenizer.encode(prompt, add_bos=True)) + len(tokenizer.encode(response, add_eos=True)) - 1
    if required > max_length:
        raise ValueError(f"Preference sequence requires {required} positions; max_length={max_length}")
    return next(pack_sft([{"prompt": prompt, "response": response}], tokenizer, max_length))


def batch_packed(samples: Iterable[PackedBatch], batch_size: int,
                 drop_last: bool = False) -> Iterator[PackedBatch]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    pending: list[PackedBatch] = []
    def combine(items: list[PackedBatch]) -> PackedBatch:
        return PackedBatch(*(np.concatenate([getattr(item, field) for item in items], axis=0)
                             for field in ("input_ids", "targets", "loss_mask", "segment_ids", "attention_mask")))
    for sample in samples:
        pending.append(sample)
        if len(pending) == batch_size:
            yield combine(pending)
            pending = []
    if pending and not drop_last:
        yield combine(pending)
