"""Tests for rules_check.py: python3 scripts/test_rules.py"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rules_check as r  # noqa: E402


class Rules(unittest.TestCase):
    def tree(self, files):
        d = tempfile.mkdtemp()
        for p, text in files.items():
            os.makedirs(os.path.dirname(f"{d}/{p}") or d, exist_ok=True)
            open(f"{d}/{p}", "w").write(text)
        return d

    def test_length_names_and_missing_files(self):
        d = self.tree({"a.rs": "x\n" * 101, "b.rs": "x\n" * 100, "Notes.md": "n", "readme.md": "r", "x_y.md": ""})
        v = r.violations(d, ["a.rs", "b.rs", "Notes.md", "readme.md", "x_y.md", "AGENTS.md"])
        self.assertEqual(v["long"], {"a.rs": 101})
        self.assertEqual(v["names"], ["Notes.md", "x_y.md"])
        self.assertEqual(v["missing"], ["next-steps.md", "metrics.md", "test-time-report.md", "system-profile.md"])

    def test_recorded_debt_may_shrink_but_not_grow(self):
        base = {"long": {"a.rs": 300}, "names": ["Old.md"], "missing": ["metrics.md"]}
        found = {"long": {"a.rs": 250, "n.rs": 101}, "names": ["Old.md", "New.md"], "missing": ["metrics.md", "readme.md"]}
        msgs = r.new_problems(found, base)
        self.assertEqual(len(msgs), 3)
        self.assertTrue(any("n.rs: 101" in m for m in msgs))
        self.assertTrue(any("New.md" in m for m in msgs))
        self.assertTrue(any("readme.md" in m for m in msgs))
        grown = r.new_problems({"long": {"a.rs": 301}, "names": [], "missing": []}, base)
        self.assertIn("recorded 300", grown[0])
        self.assertEqual(r.new_problems({"long": {"a.rs": 300}, "names": [], "missing": []}, base), [])

    def test_a_page_needs_one_diagram_per_200_words(self):
        import rules_docs as rd
        text = "word " * 450
        self.assertEqual(rd.diagrams_missing(text), 2)
        self.assertEqual(rd.diagrams_missing(text + "\n```mermaid\ngraph TD\n```\n"), 1)
        self.assertEqual(rd.diagrams_missing("short"), 0)
        d = self.tree({"a.md": text, "docs/logs/old.md": text})
        self.assertEqual(rd.missing_diagrams(d, ["a.md", "docs/logs/old.md", "gone.md"]), {"a.md": 2})

    def test_baseline_round_trip_and_missing_baseline(self):
        d = self.tree({})
        self.assertEqual(r.load_baseline(d), {"long": {}, "names": [], "missing": [], "diagrams": {}})
        os.makedirs(f"{d}/scripts")
        open(f"{d}/{r.BASELINE}", "w").write('{"long": {"a.rs": 9}, "names": [], "missing": []}')
        self.assertEqual(r.load_baseline(d)["long"], {"a.rs": 9})


if __name__ == "__main__":
    unittest.main()
