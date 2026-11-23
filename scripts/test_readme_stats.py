"""Tests for readme_stats.py: python3 scripts/test_readme_stats.py"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import readme_stats as rs  # noqa: E402

METRICS = """after `abc1234` plus
**12** `#[test]` functions counted
Rust: **500** lines of production code and **300** lines of test code
| lines | 90 | 100 | 10 | 90.00% |
| functions | 8 | 10 | 2 | 80.00% |
| **total** | PASS | 4m 01s |
"""
README = "# T\n\ntext\n\n<!-- stats:begin -->\nold\n<!-- stats:end -->\n\nafter\n"


class Stats(unittest.TestCase):
    def test_figures_are_read_and_missing_ones_are_left_out(self):
        f = rs.facts(METRICS)
        self.assertEqual((f["tests"], f["production"], f["lines"], f["gate"]), ("12", "500", "90.00", "PASS in 4m 01s"))
        self.assertNotIn("regions", f)
        self.assertNotIn("| region coverage", rs.block(METRICS))
        self.assertIn("| line coverage | 90.00% |", rs.block(METRICS))

    def test_only_the_marked_block_is_replaced_and_a_page_without_markers_is_untouched(self):
        out = rs.apply(README, METRICS)
        self.assertTrue(out.startswith("# T\n\ntext\n\n") and out.endswith("\n\nafter\n"))
        self.assertNotIn("old", out)
        self.assertEqual(rs.apply(out, METRICS), out)
        self.assertEqual(rs.apply("no markers", METRICS), "no markers")

    def test_main_writes_the_file_once(self):
        d = tempfile.mkdtemp()
        self.assertEqual(rs.main([d]), 0)  # nothing to do without the files
        open(f"{d}/readme.md", "w").write(README)
        open(f"{d}/metrics.md", "w").write(METRICS)
        self.assertEqual(rs.main([d]), 0)
        first = open(f"{d}/readme.md").read()
        self.assertIn("`#[test]` functions | 12", first)
        self.assertEqual(rs.main([d]), 0)
        self.assertEqual(open(f"{d}/readme.md").read(), first)


if __name__ == "__main__":
    unittest.main()
