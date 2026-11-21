"""Tests for dup_check.py: python3 scripts/test_dup_check.py"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dup_check as d  # noqa: E402

BLOCK = "\n".join(f"let value_number_{i} = compute_something({i});" for i in range(10)) + "\n"


class Dups(unittest.TestCase):
    def test_only_meaningful_lines_count(self):
        m = d.meaningful("use std::io;\n\n// comment\n    let long_enough_line = 1;\n}\nshort\n")
        self.assertEqual(m, [(4, "let long_enough_line = 1;")])

    def test_a_repeated_run_is_counted_once_per_extra_copy_and_unique_code_is_clean(self):
        per = d.duplicates({"a.rs": BLOCK, "b.rs": "fn first() {}\n" + BLOCK, "c.rs": "\n".join(f"let different_{i} = other({i});" for i in range(10))})
        self.assertEqual(per["b.rs"], 3)
        self.assertEqual(per["a.rs"], 0)
        self.assertEqual(per["c.rs"], 0)
        self.assertEqual(d.duplicates({"a.rs": BLOCK}), {})

    def test_short_files_have_no_windows(self):
        self.assertEqual(d.windows("a.rs", "let one_line_only = 1;\n"), [])


if __name__ == "__main__":
    unittest.main()
