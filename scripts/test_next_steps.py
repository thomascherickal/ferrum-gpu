"""Tests for next_steps.py: python3 scripts/test_next_steps.py"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import next_steps as n  # noqa: E402

METRICS = "| lines | 20888 | 21090 | 202 | 99.04% |\n| file | missed lines | of | line coverage |\n| ferrum_a/src/x.rs | 24 | 360 | 93.33% |\n"


def inp(**kw):
    base = dict(coverage=(99.04, 202), least=[("a.rs", "24")], minutes=3.2, gate=("FAIL", ["xe"]), long={"big.rs": 900, "mid.rs": 300},
                dups=4, profile=True, metrics=True, readme=True, unmerged=[("topic", 3, "feat: x")], dirty=True, backlog=["one thing"],
                branch="b", head="abc1234")
    base.update(kw)
    return base


class Steps(unittest.TestCase):
    def test_readers(self):
        self.assertEqual(n.coverage(METRICS), (99.04, 202))
        self.assertIsNone(n.coverage("nothing"))
        self.assertEqual(n.least_covered(METRICS), [("ferrum_a/src/x.rs", "24")])
        self.assertEqual(n.suite_minutes("**2910 tests**, **3.2 minutes** in total"), 3.2)
        self.assertIsNone(n.suite_minutes(""))
        self.assertEqual(n.gate_state("fmt rc=0 1\nxe rc=1 2\ncov rc=2 3\n", "FAIL\n"), ("FAIL", ["xe", "cov"]))
        self.assertEqual(n.gate_state("", ""), ("-", []))

    def test_a_failing_repo_lists_every_open_item_in_order(self):
        md = n.build(inp())
        for needle in ["[ ] 1. Tests pass", "99.04%", "[x] 3.", "[ ] 4. No file over 100 lines (2 over)", "[ ] 5.", "failed: xe",
                       "## Plan in phases", "### [ ] Phase 1", "**(current)**", "Merge `topic` (3 commits", "one thing",
                       "uncommitted changes", "Fix what it reports (failed: xe)"]:
            self.assertIn(needle, md)
        self.assertLess(md.index("Phase 1"), md.index("Phase 2"))
        self.assertEqual(md.count("**(current)**"), 1)

    def test_a_clean_repo_is_all_checked_and_points_at_the_plan(self):
        md = n.build(inp(coverage=(100.0, 0), gate=("PASS", []), long={}, dups=0, unmerged=[], dirty=False, backlog=[]))
        self.assertNotIn("- [ ] ", md)
        self.assertIn("### [x] Phase 5", md)
        self.assertIn("Pick the next item from docs/matcher-plan.md", md)
        self.assertIn("### [ ] Phase 6: housekeeping **(current)**", md)

    def test_missing_reports_are_said_not_guessed(self):
        md = n.build(inp(coverage=None, minutes=None, profile=False, metrics=False, readme=False, gate=("-", []), least=[]))
        self.assertIn("(no run recorded)", md)
        self.assertIn("(no report yet)", md)
        self.assertIn("no run recorded yet", md)

    def test_main_writes_once_and_inputs_survive_a_bare_directory(self):
        d = tempfile.mkdtemp()
        self.assertEqual(n.main([d]), 0)
        first = open(f"{d}/next-steps.md").read()
        self.assertEqual(n.main([d]), 0)
        self.assertEqual(open(f"{d}/next-steps.md").read(), first)
        self.assertEqual(n.read(d, "absent"), "")
        self.assertEqual(n.unmerged(d), [])


if __name__ == "__main__":
    unittest.main()
