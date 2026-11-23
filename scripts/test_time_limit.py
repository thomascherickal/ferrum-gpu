"""Tests for time_limit.py: python3 scripts/test_time_limit.py"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runner  # noqa: E402
import time_limit as t  # noqa: E402
import time_limit_core as c  # noqa: E402


def artifact(path, test=True, exe="x"):
    return json.dumps({"reason": "compiler-artifact", "profile": {"test": test}, "executable": exe,
                       "manifest_path": path})


class Core(unittest.TestCase):
    def test_only_test_executables_are_taken_once_each_with_their_crate_directory(self):
        lines = [artifact("/w/a/Cargo.toml", exe="/t/a-1"), artifact("/w/a/Cargo.toml", exe="/t/a-1"),
                 artifact("/w/b/Cargo.toml", test=False, exe="/t/b"), artifact("/w/c/Cargo.toml", exe=None),
                 "not json", json.dumps({"reason": "build-finished"})]
        self.assertEqual(c.parse_artifacts(lines), [("/w/a", "/t/a-1")])

    def test_limits_are_whole_seconds_and_interop_has_its_own(self):
        self.assertEqual([c.parse_limit(x) for x in ["0", "600", "", "1.5", "-1", "x"]], [0, 600, None, None, None, None])
        self.assertEqual(c.limit_for("/t/interop-abc", 0, 600), 600)
        self.assertEqual(c.limit_for("/t/api-abc", 0, 600), 0)

    def test_results_are_told_apart(self):
        self.assertEqual([c.classify(0, 0), c.classify(1, 0), c.classify(None, 60), c.classify(1, 60)],
                         ["pass", "fail", "hang", "fail"])

    def test_tests_are_read_from_the_terse_listing(self):
        self.assertEqual(c.listed_tests("a::b: test\nc: bench\nd: test\n"), ["a::b", "d"])

    def test_the_summary_lists_the_slowest_and_names_what_went_wrong(self):
        times = [(i / 10, "bin", f"t{i}") for i in range(10)]
        text = c.summary(times, 0, 600, {"pass": 8, "fail": 1, "hang": 1}, ["bad"], ["slow"])
        self.assertIn("RESULT: 8 passed, 1 failed, 1 over their limit", text)
        self.assertIn("0.900000000 bin t9", text)
        self.assertNotIn("0.100000000 bin t1", text)
        self.assertEqual(text[-2:], ["FAILED:\n  bad", "OVER THE LIMIT:\n  slow"])


class Running(unittest.TestCase):
    def test_a_test_that_passes_fails_or_overruns_gets_the_right_status(self):
        py = sys.executable
        here = os.path.dirname(os.path.abspath(__file__))
        self.assertEqual(runner.run([py, "-c", "pass"], cwd=here), 0)
        self.assertEqual(runner.run([py, "-c", "raise SystemExit(3)"], cwd=here), 3)
        self.assertIsNone(runner.run([py, "-c", "import time; time.sleep(30)"], timeout=1))
        self.assertEqual(runner.run(["no-such-program-ferrum"]), runner.NOT_FOUND)

    def test_the_limits_are_validated_before_anything_is_built(self):
        os.environ["INTEROP_LIMIT"] = "soon"
        try:
            self.assertEqual(t.main(["0"]), 2)
        finally:
            del os.environ["INTEROP_LIMIT"]
        self.assertEqual(t.main(["-3"]), 2)


if __name__ == "__main__":
    unittest.main()
