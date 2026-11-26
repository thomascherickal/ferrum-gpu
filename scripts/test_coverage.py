"""Tests for coverage.py and coverage_plan.py: python3 scripts/test_coverage.py"""
import hashlib
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coverage as cov  # noqa: E402
import coverage_plan as p  # noqa: E402


class Plan(unittest.TestCase):
    def test_the_layout_hash_ignores_order_and_changes_with_the_file_set(self):
        self.assertEqual(p.layout_hash(["b.rs", "a.rs"]), hashlib.sha1(b"a.rs\nb.rs\n").hexdigest())
        self.assertNotEqual(p.layout_hash(["a.rs"]), p.layout_hash(["a.rs", "b.rs", "b.rs\na.rs"]))  # plus the mocked compiler version

    def test_the_two_runs_and_the_report_commands(self):
        self.assertEqual(p.run_one()[:6], ["cargo", "llvm-cov", "nextest", "--workspace", "--no-report", "-E"])
        self.assertIn("not binary(interop)", p.run_one())
        self.assertIn("package(ferrum_gpu_core)", p.run_two()[-1])
        self.assertEqual(p.summary_cmd("")[-1], "--summary-only")
        self.assertEqual(p.summary_cmd("100")[-2:], ["--fail-under-lines", "100"])
        self.assertEqual(p.lcov_cmd()[-1], p.LCOV)
        self.assertEqual(p.metrics_cmd("")[-2:], ["--floor", "none"])
        self.assertEqual(p.metrics_cmd("100")[-2:], ["--floor", "100"])
        self.assertEqual(p.timing_line("no-adapter", 12.9), "no-adapter 12s")


class Main(unittest.TestCase):
    def run_main(self, summary_rc=0, lcov_rc=0, floor="100"):
        d = tempfile.mkdtemp()
        calls = []

        def record(kind, rc=0):
            return lambda cmd, *a, **k: calls.append((kind, cmd, a, k)) or rc

        names = {"TIMING": f"{d}/t.txt", "LAYOUT": f"{d}/layout", "SUMMARY": f"{d}/s.txt"}
        with mock.patch.multiple(p, **names), mock.patch.dict(os.environ, {"FLOOR": floor}), \
                mock.patch.object(cov.runner, "run", lambda cmd, *a, **k: calls.append(("run", cmd, a, k)) or
                                  (lcov_rc if "lcov_check.py" in cmd[-2] else 0)), \
                mock.patch.object(cov.runner, "run_quiet", record("quiet")), \
                mock.patch.object(cov.runner, "run_tee", record("tee", summary_rc)), \
                mock.patch.object(cov.runner, "captured", lambda cmd: "b.rs\na.rs\n"):
            status = cov.main([])
        return status, calls, d

    def test_a_clean_run_runs_both_stages_then_reports(self):
        status, calls, d = self.run_main()
        self.assertEqual(status, 0)
        cargo = [c for k, c, *_ in calls if k == "run" and c[0] == "cargo"]
        self.assertEqual(cargo, [p.run_one(), p.run_two()])
        envs = [a[0] for k, c, a, _ in calls if k == "run" and a]
        self.assertIn({"FERRUM_GPU_ADAPTER": "no-such-adapter"}, envs)
        self.assertEqual(open(f"{d}/layout").read().strip(), p.layout_hash(["a.rs", "b.rs", "b.rs\na.rs"]))  # plus the mocked compiler version
        lines = open(f"{d}/t.txt").read().splitlines()
        self.assertEqual([l.split()[0] for l in lines], ["clean-profraw", "clean-workspace-artifacts",
                                                         "default-adapter", "no-adapter", "total"])

    def test_a_floor_miss_or_an_uncovered_line_fails_the_run_after_everything_was_recorded(self):
        self.assertEqual(self.run_main(summary_rc=2)[0], 1)
        status, calls, d = self.run_main(lcov_rc=1)
        self.assertEqual(status, 1)
        self.assertTrue(any("--coverage" in c for _, c, *_ in calls))


if __name__ == "__main__":
    unittest.main()
