import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from rawllm.runtime import RunLease, StopRequest, reconcile_log, supervise

ROOT = Path(__file__).resolve().parents[1]


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def run_child(self, code, **kwargs):
        return supervise([sys.executable, "-u", "-c", code], cwd=ROOT,
                         log_path=self.root / "child.log", **kwargs)

    def test_run_lease_excludes_writers_and_survives_stale_metadata(self):
        with RunLease(self.root):
            with self.assertRaisesRegex(RuntimeError, "owns"):
                with RunLease(self.root):
                    self.fail("Two writers entered")
        with RunLease(self.root):
            self.assertTrue((self.root / ".run.lock").is_file())

    def test_kernel_releases_lease_when_owner_is_killed(self):
        code = "from rawllm.runtime import RunLease; import sys,time\nwith RunLease(sys.argv[1]):\n print('ready',flush=True)\n time.sleep(10)"
        child = subprocess.Popen([sys.executable, "-u", "-c", code, str(self.root)], cwd=ROOT, stdout=subprocess.PIPE)
        try:
            self.assertEqual(child.stdout.readline(), b"ready\n")
            with self.assertRaises(RuntimeError):
                with RunLease(self.root):
                    self.fail("Live owner was ignored")
            child.kill()
            child.wait(timeout=2)
            with RunLease(self.root):
                pass
        finally:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=2)
            child.stdout.close()

    def test_supervisor_success_exit_and_single_thread_environment(self):
        result = self.run_child("import os;print(os.environ['OPENBLAS_NUM_THREADS'])")
        self.assertEqual(result["reason"], "completed")
        self.assertEqual(result["returncode"], 0)
        self.assertEqual((self.root / "child.log").read_text(), "1\n")

    def test_supervisor_child_failure(self):
        result = self.run_child("raise SystemExit(19)")
        self.assertEqual(result["reason"], "failed")
        self.assertEqual(result["returncode"], 19)

    def test_supervisor_deadline_escalates_uncooperative_process(self):
        result = self.run_child("import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(10)",
                                max_seconds=.2, grace_seconds=.05)
        self.assertEqual(result["reason"], "deadline")
        self.assertEqual(result["returncode"], -signal.SIGKILL)
        self.assertLess(result["seconds"], 2)

    def test_supervisor_caps_output(self):
        result = self.run_child("import os,time;os.write(1,b'x'*1000000);time.sleep(10)", max_output_bytes=321, grace_seconds=.05)
        self.assertEqual(result["reason"], "output_limit")
        self.assertEqual((self.root / "child.log").stat().st_size, 321)

    def test_supervisor_cleans_descendants_when_leader_exits(self):
        ready, stopped = self.root / "ready", self.root / "stopped"
        child_code = ("import signal,time,pathlib,sys\n"
                      f"def stop(*a):\n pathlib.Path({str(stopped)!r}).write_text('stopped')\n sys.exit(0)\n"
                      "signal.signal(signal.SIGTERM,stop)\n"
                      f"pathlib.Path({str(ready)!r}).write_text('ready')\n"
                      "time.sleep(10)")
        code = ("import subprocess,sys,time,pathlib\n"
                f"subprocess.Popen([sys.executable,'-c',{child_code!r}])\n"
                f"while not pathlib.Path({str(ready)!r}).exists(): time.sleep(.005)\n")
        result = self.run_child(code, max_seconds=2, grace_seconds=.2)
        self.assertEqual(result["reason"], "completed")
        self.assertTrue(stopped.exists())
        self.assertLess(result["seconds"], 2)

    def test_stop_handler_is_cooperative_and_restored(self):
        previous = signal.getsignal(signal.SIGTERM)
        with StopRequest() as stop:
            os.kill(os.getpid(), signal.SIGTERM)
            self.assertTrue(stop.requested)
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)

    def test_log_reconciles_only_uncommitted_tail(self):
        path = self.root / "training.jsonl"
        path.write_text('{"step":1}\n{"step":2}\n{"step":3}\n{"step":')
        reconcile_log(path, 2)
        self.assertEqual(path.read_text(), '{"step":1}\n{"step":2}\n')
        path.write_text('{"step":1}\nbroken\n{"step":2}\n')
        with self.assertRaises(ValueError):
            reconcile_log(path, 2)

    def test_training_sigterm_commits_boundary_and_resumes(self):
        run = self.root / "run"
        command = [sys.executable, "-u", "train.py", "pretrain", "--output", str(run),
                   "--steps", "100000", "--sequence-length", "8", "--max-seconds", "10"]
        child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            record = json.loads(child.stdout.readline())
            self.assertGreater(record["step"], 0)
            child.send_signal(signal.SIGTERM)
            stdout, stderr = child.communicate(timeout=5)
            self.assertEqual(child.returncode, 0, stderr.decode())
            status = json.loads((run / "status.json").read_text())
            self.assertEqual(status["status"], "stopped")
            before = json.loads((run / "checkpoint/manifest.json").read_text())["counters"]["step"]
            resumed = subprocess.run(command + ["--resume", "--steps", str(before + 1)], cwd=ROOT,
                                     capture_output=True, timeout=5)
            self.assertEqual(resumed.returncode, 0, resumed.stderr.decode())
            after = json.loads((run / "checkpoint/manifest.json").read_text())["counters"]["step"]
            self.assertEqual(after, before + 1)
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate(timeout=2)
            child.stdout.close()
            child.stderr.close()


if __name__ == "__main__":
    unittest.main()
