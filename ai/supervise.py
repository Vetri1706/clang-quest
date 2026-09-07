"""Run a trusted framework command with a time/output budget and process cleanup."""
import argparse
from pathlib import Path
import sys
from rawllm.runtime import RunLease, atomic_json, supervise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New watchdog report directory")
    parser.add_argument("--seconds", type=float, default=90)
    parser.add_argument("--output-bytes", type=int, default=2 * 1024**2)
    parser.add_argument("command", choices=["train", "zero_train", "zero_corpus", "distributed_train", "distributed_llm3d", "distributed_demo", "verify"])
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    with RunLease(args.output):
        result = supervise([sys.executable, "-u", str(root / (args.command + ".py")), *args.arguments],
                           cwd=root, log_path=args.output / "process.log", max_seconds=args.seconds,
                           max_output_bytes=args.output_bytes)
        atomic_json(args.output / "result.json", result)
    print(__import__("json").dumps(result))
    return 0 if result["reason"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
