"""Bounded numerical artifacts for trusted local checkpoint directories.

Hashes detect corruption, not authenticity. Parent directories must be owned by
the operator: these helpers are not a sandbox against concurrent file mutation.
No pickle, object dtype, duplicate ZIP member, or symbolic-link file is accepted.
"""
from contextlib import contextmanager
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import struct
import zipfile

import numpy as np


@contextmanager
def regular_file(path, max_bytes):
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("A positive byte limit is required")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
            raise ValueError("Artifact is not a regular file within the byte limit")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            yield handle
    finally:
        os.close(descriptor)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object field")
        result[key] = value
    return result


def read_json(path, max_bytes=1024 * 1024):
    def reject_constant(value):
        raise ValueError("Non-finite JSON constant")
    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("Non-finite JSON number")
        return parsed
    with regular_file(path, max_bytes) as handle:
        raw = handle.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError("JSON artifact grew beyond the byte limit")
    try:
        return json.loads(raw, object_pairs_hook=_unique_object, parse_constant=reject_constant, parse_float=finite_float)
    except (RecursionError, UnicodeError) as error:
        raise ValueError("Invalid JSON artifact") from error


def _check_zip_directory(handle, count):
    """Bound the central directory before ZipFile parses it into Python objects."""
    size = os.fstat(handle.fileno()).st_size
    start = max(0, size - 65557)
    handle.seek(start)
    tail = handle.read(65557)
    index = len(tail)
    while True:
        index = tail.rfind(b"PK\x05\x06", 0, index)
        if index < 0:
            raise ValueError("Missing ZIP end record")
        if index + 22 <= len(tail):
            fields = struct.unpack_from("<4s4H2LH", tail, index)
            if index + 22 + fields[-1] == len(tail):
                break
    _, disk, directory_disk, disk_entries, entries, length, offset, _ = fields
    boundary = start + index
    if disk or directory_disk or disk_entries != entries:
        raise ValueError("Split ZIP archives are unsupported")
    if entries == 0xFFFF or length == 0xFFFFFFFF or offset == 0xFFFFFFFF:
        if boundary < 20:
            raise ValueError("Missing ZIP64 locator")
        handle.seek(boundary - 20)
        magic, disk, location, disks = struct.unpack("<4sLQL", handle.read(20))
        if magic != b"PK\x06\x07" or disk != 0 or disks != 1 or location + 56 > boundary - 20:
            raise ValueError("Invalid ZIP64 locator")
        handle.seek(location)
        fields64 = struct.unpack("<4sQ2H2L4Q", handle.read(56))
        magic, record_size, _, _, disk, directory_disk, disk_entries, entries, length, offset = fields64
        if magic != b"PK\x06\x06" or not 44 <= record_size <= 1024 or location + 12 + record_size != boundary - 20:
            raise ValueError("Invalid ZIP64 end record")
        if disk or directory_disk or disk_entries != entries:
            raise ValueError("Split ZIP64 archives are unsupported")
        boundary = location
    if entries != count or length > 4096 * count or length < 46 * count or offset + length != boundary:
        raise ValueError("ZIP central directory exceeds its expected inventory or byte bound")
    handle.seek(0)


def read_npz(path, expected, *, sha256=None, max_bytes=512 * 1024**2, finite=True, select=None):
    """Read an exact mapping of key -> (shape, dtype or tuple of allowed dtypes).

    Inspect every NPY header and uncompressed extent before allocating any array.
    The byte limit covers summed uncompressed data, with 4 KiB/member header and
    ZIP overhead allowance. Returned arrays consume their aggregate data bytes;
    parsing and decompression have additional bounded temporary storage.
    """
    if not expected or len(expected) > 16384:
        raise ValueError("Invalid array inventory size")
    selected = set(expected) if select is None else set(select)
    if not selected.issubset(expected):
        raise ValueError("Selected arrays are absent from the schema")
    schema = {}
    total = 0
    for key, (shape, dtypes) in expected.items():
        if not isinstance(key, str) or not key or len(key.encode("utf8")) > 512 or "/" in key or "\\" in key:
            raise ValueError("Invalid array name")
        shape = tuple(shape)
        if len(shape) > 16 or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError("Invalid expected array shape")
        dtypes = dtypes if isinstance(dtypes, tuple) else (dtypes,)
        allowed = tuple(np.dtype(d) for d in dtypes)
        if not allowed or any(d.hasobject or d.fields or d.subdtype or d.kind not in "fiub" for d in allowed):
            raise ValueError("Only plain numeric and boolean arrays are supported")
        count = math.prod(shape)
        total += count * max(d.itemsize for d in allowed)
        schema[key + ".npy"] = (shape, allowed)
    if total > max_bytes:
        raise MemoryError("Requested archive arrays exceed the allocation limit")
    overhead = 4096 * len(schema) + 65536
    with regular_file(path, total + overhead) as handle:
        if sha256 is not None:
            if not isinstance(sha256, str) or len(sha256) != 64:
                raise ValueError("Invalid SHA256 digest")
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            if digest.hexdigest() != sha256:
                raise ValueError("Artifact checksum mismatch")
            handle.seek(0)
        try:
            _check_zip_directory(handle, len(schema))
            with zipfile.ZipFile(handle) as archive:
                entries = archive.infolist()
                if len(entries) != len(schema) or set(i.filename for i in entries) != set(schema):
                    raise ValueError("Archive array inventory does not match expected schema")
                if sum(i.file_size for i in entries) > total + 4096 * len(schema):
                    raise ValueError("Archive uncompressed size exceeds expected allocation")
                for entry in entries:
                    if entry.flag_bits & 1 or entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                        raise ValueError("Unsupported archive encryption or compression")
                    shape, dtypes = schema[entry.filename]
                    with archive.open(entry) as member:
                        version = np.lib.format.read_magic(member)
                        reader = {(1, 0): np.lib.format.read_array_header_1_0,
                                  (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                        if reader is None:
                            raise ValueError("Only NPY versions 1.0 and 2.0 are supported")
                        # NumPy checks max_header_size after reading the declared
                        # header, so enforce the bound before it reads anything.
                        word = member.read(2 if version == (1, 0) else 4)
                        if len(word) != (2 if version == (1, 0) else 4):
                            raise ValueError("Truncated NPY header length")
                        header_size = int.from_bytes(word, "little")
                        if header_size > 2048:
                            raise ValueError("NPY header exceeds its byte bound")
                        header = member.read(header_size)
                        if len(header) != header_size:
                            raise ValueError("Truncated NPY header")
                        actual_shape, _, dtype = reader(io.BytesIO(word + header), max_header_size=2048)
                        if actual_shape != shape or dtype not in dtypes or dtype.hasobject:
                            raise ValueError("Archive NPY shape or dtype mismatch")
                        if entry.file_size != member.tell() + math.prod(shape) * dtype.itemsize:
                            raise ValueError("Archive NPY extent mismatch")
                values = {}
                for entry in entries:
                    if entry.filename[:-4] not in selected:
                        continue
                    with archive.open(entry) as member:
                        value = np.lib.format.read_array(member, allow_pickle=False, max_header_size=2048)
                    if finite and not np.isfinite(value).all():
                        raise ValueError("Non-finite array in artifact")
                    values[entry.filename[:-4]] = value
                return values
        except (zipfile.BadZipFile, EOFError, OverflowError, struct.error) as error:
            raise ValueError("Invalid numerical archive") from error
