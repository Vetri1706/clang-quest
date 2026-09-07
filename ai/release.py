"""Build or verify a hashed source/evidence archive, excluding runtime secrets."""
import argparse
import hashlib
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parent
EXCLUDED_PARTS = {"__pycache__", ".git", ".venv", "work", ".pytest_cache"}
EXCLUDED_NAMES = {"MANIFEST.sha256", ".DS_Store", ".run.lock", ".writer.lock", ".env"}


def files():
    result = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS for part in relative.parts) or path.name in EXCLUDED_NAMES:
            continue
        if path.suffix in {".pyc", ".token", ".key", ".pem", ".partial", ".zip"}:
            continue
        if path.is_symlink():
            raise ValueError(f"Release contains a symbolic link: {relative}")
        if path.is_file():
            result.append(path)
    return sorted(result)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT.parent / (ROOT.name + ".zip"))
    args = parser.parse_args()
    paths = files()
    manifest = "".join(f"{digest(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in paths)
    manifest_path = ROOT / "MANIFEST.sha256"
    if args.check:
        if manifest_path.read_text() != manifest:
            raise ValueError("Release file inventory or SHA256 content changed")
        print(f"Verified {len(paths)} source and evidence files")
        return
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("Place release archives outside the project directory")
    manifest_path.write_text(manifest)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in [*paths, manifest_path]:
            archive.write(path, (Path(ROOT.name) / path.relative_to(ROOT)).as_posix())
    with zipfile.ZipFile(output) as archive:
        error = archive.testzip()
        if error is not None:
            raise ValueError(f"Archive validation failed: {error}")
    print(f"Created {output.name}: {len(paths) + 1} files, {output.stat().st_size} bytes; SHA256 {digest(output)}")


if __name__ == "__main__":
    main()
