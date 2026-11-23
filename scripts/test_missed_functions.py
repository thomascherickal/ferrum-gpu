"""Tests for missed_functions.py: python3 scripts/test_missed_functions.py"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import missed_functions as mf  # noqa: E402


def fn(file, line, count, name="f"):
    return {"filenames": [f"/home/u/repo/{file}"], "regions": [[line, 1, line + 1, 2, count, 0, 0, 0]], "count": count, "name": name}


class Missed(unittest.TestCase):
    def test_instances_of_a_generic_are_merged_and_only_all_zero_counts_as_missed(self):
        data = {"data": [{"functions": [fn("a.rs", 3, 0), fn("a.rs", 3, 5), fn("a.rs", 9, 0), fn("b/tests/t.rs", 1, 0), fn("x.rs", 2, 7)]}]}
        rows = mf.missed(data, "repo/")
        self.assertEqual([(r[0], r[1]) for r in rows], [("a.rs", 9), ("b/tests/t.rs", 1)])
        self.assertEqual([(r[0], r[1]) for r in mf.production(rows)], [("a.rs", 9)])

    def test_usage_without_arguments(self):
        self.assertEqual(mf.main([]), 2)


if __name__ == "__main__":
    unittest.main()
