"""Run complete local tests and audit imports; write an inspectable report."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import unittest


def main():
    root = Path(__file__).resolve().parent
    os.chdir(root)
    forbidden = {"torch", "tensorflow", "jax", "transformers", "datasets", "peft", "trl", "langchain", "accelerate", "huggingface_hub", "flax", "optax"}
    violations = []
    files = sorted(p for p in root.rglob("*.py") if not {"__pycache__", ".venv"}.intersection(p.relative_to(root).parts))
    local_modules = {p.stem for p in root.glob("*.py")} | {p.name for p in root.iterdir() if p.is_dir()}
    third_party = set()
    source_digest = hashlib.sha256()
    for path in files:
        source_digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in names:
                prefix = name.split(".")[0]
                if not (isinstance(node, ast.ImportFrom) and node.level) and prefix not in sys.stdlib_module_names | local_modules:
                    third_party.add(prefix)
                if name.split(".")[0] in forbidden:
                    violations.append({"path": str(path.relative_to(root)), "line": node.lineno, "import": name})
    if violations:
        raise RuntimeError(json.dumps(violations))
    if third_party - {"numpy"}:
        raise RuntimeError("Unexpected third-party source imports: " + repr(sorted(third_party)))
    started = time.monotonic()
    suite = unittest.defaultTestLoader.discover(str(root / "tests"), top_level_dir=str(root))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    report = {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
              "skipped": len(result.skipped), "passed": result.wasSuccessful(),
              "seconds": time.monotonic() - started, "python_files_audited": len(files),
              "forbidden_imports": violations,
              "third_party_static_imports": sorted(third_party), "python_source_sha256": source_digest.hexdigest(),
              "python_version": platform.python_version(), "numpy_version": importlib.metadata.version("numpy"),
              "system": platform.system(), "machine": platform.machine(),
              "audit_scope": "Static imports in project Python source; no claim of sandboxing arbitrary dynamic code",
              "limitations": "Small local numerical and process tests; not production certification or GPU validation."}
    (root / "results").mkdir(exist_ok=True)
    (root / "results/verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
