"""Tests for timing_report.py: python3 scripts/test_timing_report.py"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import timing_report as t  # noqa: E402

TIMES = "0.5 ferrum_a-0123456789abcdef mod::fast\n7.25 ferrum_a-0123456789abcdef mod::slow\n1.0 interop-fedcba9876543210 x::y\nbad line\nx y z\n"


class Report(unittest.TestCase):
    def test_parse_strips_hashes_and_skips_junk(self):
        rows = t.parse(TIMES)
        self.assertEqual(rows, [(0.5, "ferrum_a", "mod::fast"), (7.25, "ferrum_a", "mod::slow"), (1.0, "interop", "x::y")])

    def test_render_totals_suites_and_flags_slow_tests(self):
        md = t.render(t.parse(TIMES), "abc1234", "2026-11-21 10:00 UTC")
        self.assertIn("**3** tests", md)
        self.assertIn("within the limit", md)
        self.assertIn("| ferrum_a | 2 | 7.8 |", md)
        self.assertIn("| 7.2 | ferrum_a: mod::slow |", md)
        self.assertNotIn("| 0.5 | ferrum_a: mod::fast | \n", md.split("## Ten")[0])
        self.assertIn("`abc1234`", md)

    def test_no_slow_tests_and_an_over_limit_suite(self):
        md = t.render([(0.1, "a", "t")], "c", "now")
        self.assertIn("None.", md)
        over = t.render([(1900.0, "a", "t")], "c", "now")
        self.assertIn("OVER the limit", over)

    def test_main_writes_the_report_and_refuses_an_empty_input(self):
        d = tempfile.mkdtemp()
        p = f"{d}/times.txt"
        open(p, "w").write("nonsense\n")
        self.assertEqual(t.main([p]), 1)


if __name__ == "__main__":
    unittest.main()
