"""Tests for gate.py and gate_steps.py: python3 scripts/test_gate.py"""
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import all_tests as a_t  # noqa: E402
import coverage as cov  # noqa: E402
import gate  # noqa: E402
import time_limit as tl  # noqa: E402
import gate_steps as s  # noqa: E402

PY = sys.executable


class Steps(unittest.TestCase):
    def test_the_steps_keep_their_names_and_order(self):
        names = [n for n, *_ in s.steps("/o")]
        self.assertEqual(names, ["fmt", "clippy", "rules", "dups", "scripts", "llvmpipe", "xe", "timelimit",
                                 "timing", "coverage", "deny", "studio", "studiodeny"])

    def test_each_step_gets_its_environment_and_directory(self):
        by = {n: (e, d) for n, _, e, d in s.steps("/o")}
        self.assertEqual(by["llvmpipe"][0], {"FERRUM_GPU_ADAPTER": "llvmpipe"})
        self.assertEqual(by["timelimit"][0], {"TIMES_OUT": "/o/test-times.txt"})
        self.assertEqual(by["coverage"][0], {"FLOOR": "100"})
        self.assertEqual(by["studio"], ({"FERRUM_GPU_ADAPTER": "no-such-adapter"}, "ferrum_gpu_studio"))

    def test_help_prints_the_description_and_runs_nothing(self):
        for module in (gate, cov, a_t, tl):
            self.assertEqual(module.main(["--help"]), 0)

    def test_the_records_read_as_before(self):
        self.assertEqual(s.step_line("fmt", 0, "12:00:01"), "fmt rc=0 12:00:01")
        self.assertEqual(s.commit_record("", "abc\n"), "abc")
        self.assertEqual(s.commit_record(" M a.rs\n", "abc"), "dirty")
        self.assertEqual(s.status_args()[:4], ["status", "--porcelain", "--", "."])
        self.assertIn(":!metrics.md", s.status_args())
        self.assertEqual([s.verdict(False), s.verdict(True)], ["PASS", "FAIL"])


class Running(unittest.TestCase):
    def test_commands_stop_at_the_first_failure_and_their_output_is_logged(self):
        d = tempfile.mkdtemp()
        log = f"{d}/step.log"
        ok, bad = [PY, "-c", "print('one')"], [PY, "-c", "import sys; print('two'); sys.exit(4)"]
        self.assertEqual(gate.run_step([ok, bad, ok], {}, None, log), 4)
        self.assertEqual(open(log).read().split(), ["one", "two"])
        self.assertEqual(gate.run_step([ok], {}, d, log), 0)

    def test_the_script_tests_run_in_order_and_the_first_failure_stops_them(self):
        d = tempfile.mkdtemp()
        os.makedirs(f"{d}/scripts")
        for name, body in [("test_a.py", "print('a')"), ("test_b.py", "raise SystemExit(1)"), ("test_c.py", "print('c')")]:
            open(f"{d}/scripts/{name}", "w").write(body + "\n")
        here = os.getcwd()
        os.chdir(d)
        try:
            self.assertEqual(gate.run_step("script_tests", {}, None, f"{d}/t.log"), 1)
        finally:
            os.chdir(here)
        self.assertEqual(open(f"{d}/t.log").read().split(), ["a"])

    def test_a_whole_run_writes_the_files_the_hooks_read(self):
        out = tempfile.mkdtemp()
        two = [("good", [[PY, "-c", "pass"]], {}, None), ("bad", [[PY, "-c", "raise SystemExit(1)"]], {}, None)]
        calls, real, sampler = [], gate.runner.run, mock.Mock()

        def fake_run(cmd, *args, **kwargs):  # the report generators only; the steps really run
            if str(cmd[1]).startswith("scripts/"):
                calls.append(cmd[1:])
                return 0
            return real(cmd, *args, **kwargs)

        with mock.patch.dict(os.environ, {"GATE_OUT": out}), mock.patch.object(s, "steps", lambda o: two), \
                mock.patch.object(gate.runner, "start", return_value=sampler), \
                mock.patch.object(gate.runner, "run", fake_run):
            self.assertEqual(gate.main([]), 0)
        lines = open(f"{out}/gate.txt").read().splitlines()
        self.assertEqual([l.split()[0:2] for l in lines], [["start", "rc=0"], ["good", "rc=0"], ["bad", "rc=1"], ["budget", "rc=0"]])
        self.assertEqual(open(f"{out}/gate.done").read(), "FAIL\n")
        self.assertTrue(open(f"{out}/commit").read().strip())
        self.assertTrue(os.path.exists(f"{out}/key"))
        self.assertIn(["scripts/metrics.py", "--gate", f"{out}/gate.txt"], calls)
        self.assertIn(["scripts/next_steps.py"], calls)
        sampler.wait.assert_called_once()

    def test_budgets_are_twenty_minutes_each_for_the_tests_and_for_coverage(self):
        ok = {"llvmpipe": 400, "xe": 400, "timelimit": 399, "coverage": 1200}
        self.assertEqual(s.over_budget(ok), [])
        slow = s.over_budget({"llvmpipe": 600, "xe": 400, "timelimit": 201, "coverage": 1201})
        self.assertEqual(len(slow), 2)
        self.assertTrue(slow[0].startswith("the test steps") and slow[1].startswith("the coverage check"))
        self.assertEqual(s.over_budget({}), [])


if __name__ == "__main__":
    unittest.main()
