"""Tests for lcov_check.py: python3 scripts/test_lcov_check.py"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lcov_check as lc  # noqa: E402

LCOV = "SF:/r/a.rs\nDA:1,5\nDA:2,0\nDA:3,0\nend_of_record\nSF:/r/target/x.rs\nDA:1,0\nend_of_record\nSF:/r/b.rs\nDA:1,1\nend_of_record\n"


class Lcov(unittest.TestCase):
    def test_zero_count_lines_are_found_and_build_output_is_ignored(self):
        miss, total = lc.uncovered(LCOV)
        self.assertEqual(miss, {"/r/a.rs": [2, 3]})
        self.assertEqual(total, 4)

    def test_exit_status_follows_the_result(self):
        d = tempfile.mkdtemp()
        bad, good = f"{d}/bad.lcov", f"{d}/good.lcov"
        open(bad, "w").write(LCOV)
        open(good, "w").write("SF:/r/b.rs\nDA:1,1\nend_of_record\n")
        self.assertEqual(lc.main([bad]), 1)
        self.assertEqual(lc.main([good]), 0)
        self.assertEqual(lc.main([]), 2)


if __name__ == "__main__":
    unittest.main()
