import hashlib
import io
import json
from pathlib import Path
import tempfile
import struct
import unittest
from unittest.mock import patch
import warnings
import zipfile

import numpy as np

from rawllm.optim import AdamW, load_checkpoint, save_checkpoint
from rawllm.safeio import read_json, read_npz
from rawllm.tensor import Parameter
from rawllm.tokenizer import ByteBPETokenizer
from rawllm.model import Config


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_numeric_fortran_compressed_and_selective_load(self):
        path = self.root / "value.npz"
        first = np.asfortranarray(np.arange(12, dtype=np.float32).reshape(3, 4))
        np.savez_compressed(path, first=first, second=np.ones(2, dtype=np.int64))
        expected = {"first": ((3, 4), np.float32), "second": ((2,), np.int64)}
        arrays = read_npz(path, expected, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), select=["first"])
        self.assertEqual(set(arrays), {"first"})
        np.testing.assert_array_equal(arrays["first"], first)

    def test_wrong_shape_rejected_before_array_allocation(self):
        path = self.root / "value.npz"
        np.savez(path, value=np.ones((8, 2), dtype=np.float32))
        with patch("numpy.lib.format.read_array", side_effect=AssertionError("allocated")):
            with self.assertRaisesRegex(ValueError, "shape or dtype"):
                read_npz(path, {"value": ((4, 4), np.float32)})

    def test_object_and_duplicate_members_rejected_before_loading(self):
        path = self.root / "object.npz"
        np.savez(path, value=np.array([{"danger": "never unpickle"}], dtype=object))
        with patch("numpy.lib.format.read_array", side_effect=AssertionError("allocated")):
            with self.assertRaises(ValueError):
                read_npz(path, {"value": ((1,), np.float32)})
        value = io.BytesIO()
        np.save(value, np.ones(1, dtype=np.float32))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("value.npy", value.getvalue())
                archive.writestr("value.npy", value.getvalue())
        with self.assertRaisesRegex(ValueError, "inventory"):
            read_npz(path, {"value": ((1,), np.float32)})

    def test_zip_expansion_and_requested_allocation_limits(self):
        path = self.root / "large.npz"
        np.savez_compressed(path, value=np.zeros(1024 * 1024, dtype=np.float32))
        with patch("numpy.lib.format.read_array", side_effect=AssertionError("allocated")):
            with self.assertRaisesRegex(ValueError, "uncompressed"):
                read_npz(path, {"value": ((1,), np.float32)})
            with self.assertRaises(MemoryError):
                read_npz(path, {"value": ((1024 * 1024,), np.float32)}, max_bytes=1000)

    def test_checksum_and_symlink_rejection(self):
        path = self.root / "array.npz"
        np.savez(path, value=np.ones(1, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "checksum"):
            read_npz(path, {"value": ((1,), np.float32)}, sha256="0" * 64)
        link = self.root / "link.npz"
        link.symlink_to(path)
        with self.assertRaises(OSError):
            read_npz(link, {"value": ((1,), np.float32)})

    def test_strict_bounded_json(self):
        path = self.root / "value.json"
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'):
            path.write_text(text)
            with self.assertRaises(ValueError):
                read_json(path)
        path.write_text('"0123456789"')
        with self.assertRaises(ValueError):
            read_json(path, max_bytes=4)

    def test_declared_header_length_rejected_before_numpy_reads_it(self):
        path = self.root / "header.npz"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("value.npy", b"\x93NUMPY\x02\x00" + struct.pack("<I", 2**30) + b" " * 100)
        with patch("numpy.lib.format.read_array_header_2_0", side_effect=AssertionError("header allocated")):
            with self.assertRaisesRegex(ValueError, "header exceeds"):
                read_npz(path, {"value": ((10,), np.float32)})

    def test_directory_inventory_checked_before_zipfile_allocation(self):
        path = self.root / "directory.npz"
        np.savez(path, value=np.ones(1, dtype=np.float32))
        content = bytearray(path.read_bytes())
        content[-14:-10] = struct.pack("<HH", 10000, 10000)
        path.write_bytes(content)
        with patch("rawllm.safeio.zipfile.ZipFile", side_effect=AssertionError("directory allocated")):
            with self.assertRaisesRegex(ValueError, "central directory"):
                read_npz(path, {"value": ((1,), np.float32)})

    def test_bpe_expansion_bomb_and_noninteger_merges(self):
        pairs = [(100, 100)]
        for index in range(40):
            previous = 259 + index
            pairs.append((previous, previous))
        with self.assertRaisesRegex(ValueError, "byte limit"):
            ByteBPETokenizer(pairs)
        with self.assertRaises(ValueError):
            ByteBPETokenizer([(100.5, 100)])

    def test_nonfinite_optimizer_candidate_is_atomic_for_all_parameters(self):
        params = {"a": Parameter(np.ones(3, dtype=np.float32)), "b": Parameter(np.ones(3, dtype=np.float32))}
        optimizer = AdamW(params)
        before = optimizer.state_dict()
        params["a"].grad = np.ones(3, dtype=np.float32)
        params["b"].grad = np.full(3, 1e30, dtype=np.float32)
        with self.assertRaisesRegex(FloatingPointError, "no optimizer state"):
            optimizer.step()
        self.assertEqual(optimizer.step_count, 0)
        self.assertEqual(optimizer.steps, before["steps"])
        for name in params:
            np.testing.assert_array_equal(params[name].data, np.ones(3))
            for group in ("master", "m", "v"):
                np.testing.assert_array_equal(getattr(optimizer, group)[name], before[group][name])

    def test_visible_dtype_overflow_does_not_commit_fp32_master(self):
        parameter = Parameter(np.ones(2, dtype=np.float16))
        optimizer = AdamW({"w": parameter}, lr=1e6)
        parameter.grad = np.ones(2, dtype=np.float32)
        with self.assertRaises(FloatingPointError):
            optimizer.step()
        np.testing.assert_array_equal(optimizer.master["w"], np.ones(2))
        self.assertEqual(optimizer.steps["w"], 0)

    def test_invalid_counters_do_not_mutate_checkpoint_target(self):
        param = Parameter(np.ones(2, dtype=np.float32))
        optimizer = AdamW({"w": param})
        rng = np.random.default_rng(4)
        save_checkpoint(self.root, {"w": param}, optimizer, rng, {"step": 0})
        path = self.root / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["counters"] = [1, 2]
        path.write_text(json.dumps(manifest))
        param.data[:] = 2
        with self.assertRaisesRegex(ValueError, "counters"):
            load_checkpoint(self.root, {"w": param}, optimizer, rng)
        np.testing.assert_array_equal(param.data, [2, 2])

    def test_json_array_dtype_expansion_rejected_before_restore(self):
        parameter = Parameter(np.ones(2, dtype=np.float32))
        optimizer = AdamW({"w": parameter})
        rng = np.random.default_rng(4)
        save_checkpoint(self.root, {"w": parameter}, optimizer, rng, {"step": 0})
        path = self.root / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["counters"] = {"x": {"__ndarray__": ["x"], "dtype": "S1000000000"}}
        path.write_text(json.dumps(manifest))
        parameter.data[:] = 2
        with self.assertRaisesRegex(ValueError, "numerical dtype"):
            load_checkpoint(self.root, {"w": parameter}, optimizer, rng)
        np.testing.assert_array_equal(parameter.data, [2, 2])

    def test_boolean_dimensions_and_fractional_optimizer_steps_rejected(self):
        with self.assertRaises(ValueError):
            Config(layers=True)
        with self.assertRaises(ValueError):
            Config(tie_embeddings="false")
        optimizer = AdamW({"w": Parameter(np.ones(1, dtype=np.float32))})
        for step in (True, 1.5):
            state = optimizer.state_dict()
            state["step_count"] = step
            with self.assertRaises(ValueError):
                optimizer.load_state_dict(state)
            self.assertEqual(optimizer.step_count, 0)


if __name__ == "__main__":
    unittest.main()
