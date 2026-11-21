"""Tests for generated_diagrams.py, docs_outline.py and docs_diagrams.py: python3 scripts/test_generated_diagrams.py"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import docs_diagrams as dd  # noqa: E402
import docs_outline as do  # noqa: E402
import generated_diagrams as g  # noqa: E402
import rules_docs as rd  # noqa: E402

PAGE = "# T\n\nintro\n\n## Alpha\n\n" + "word " * 450 + "\n\n- **Bold term** one\n- item two\n\n### Sub\n\n## Beta\n\nshort\n"


class Diagrams(unittest.TestCase):
    def test_pie_and_trend_and_flow(self):
        self.assertIn('"Rust" : 10', g.pie("Lines", [("Rust", 10), ("none", 0)]))
        self.assertEqual(g.pie("Lines", [("none", 0)]), "")
        self.assertIn("bar [1, 2]", g.trend("CPU", ["a", "b"], [1, 2], "percent"))
        self.assertEqual(g.trend("CPU", [], [], "percent"), "")
        flow = g.checklist_flow(["[x] 1. Tests pass (ok)", "[ ] 2. Coverage is 100% (99%)"])
        self.assertIn(":::met", flow)
        self.assertIn(":::open", flow)
        self.assertIn("s0 --> s1", flow)

    def test_outline_reads_sections_labels_and_words(self):
        secs = do.sections(PAGE)
        self.assertEqual([s[0] for s in secs], ["Alpha", "Beta"])
        self.assertGreater(do.words(secs[0][1]), 440)
        self.assertEqual(do.labels(secs[0][1])[:3], ["Sub", "Bold term", "Bold term one"])
        self.assertEqual(do.clean("`ferrum_gguf` (the **file** reader) with a very long tail here"), "ferrum_gguf the file reader with")
        self.assertEqual([s[0] for s in do.sections("### Only\ntext")], ["Only"])

    def test_pages_get_exactly_the_diagrams_they_lack_and_no_more(self):
        self.assertEqual(rd.diagrams_missing(PAGE), 2)
        done = dd.apply(PAGE)
        self.assertEqual(rd.diagrams_missing(done), 0)
        self.assertEqual(dd.apply(done), done)
        self.assertEqual(dd.plan("short page"), {})
        self.assertIn("classDef root", dd.diagram("T", ["a", "b"]))


if __name__ == "__main__":
    unittest.main()
