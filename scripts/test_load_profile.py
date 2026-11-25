"""Tests for load_profile.py: python3 scripts/test_load_profile.py"""
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import load_profile as lp  # noqa: E402
import system_profile as sp  # noqa: E402

LINE = "LOAD name=json-server clients=48 requests=192 ok=190 refused=2 seconds=0.15 rps=1263 p50_ms=31.9 " \
       "p95_ms=54.8 p99_ms=55.8 max_ms=56.0"


class LoadProfile(unittest.TestCase):
    def test_load_lines_become_table_rows(self):
        rows = lp.parse("noise\ntest a::b ... " + LINE + "\nLOAD name=x\n")
        self.assertEqual((len(rows), rows[0]["p95_ms"], rows[1]["name"]), (2, "54.8", "x"))
        b = lp.block(rows, "2026-11-25 10:00", "abc", "debug")
        self.assertIn("| json-server | 48 | 192 | 190 | 2 | 0.15 | 1263 | 31.9 | 54.8 | 55.8 | 56.0 |", b)
        self.assertIn("| x | - |", b)
        self.assertTrue(b.startswith(lp.LBEGIN) and b.endswith(lp.LEND))

    def test_the_section_is_replaced_in_place_and_survives_a_gate_row(self):
        doc = lp.splice("# System profile\n", "<!-- load:begin -->A<!-- load:end -->")
        self.assertTrue(doc.endswith("<!-- load:end -->\n"))
        doc = lp.splice(doc + "tail\n", "<!-- load:begin -->B<!-- load:end -->")
        self.assertIn("B<!-- load:end -->\ntail", doc)
        self.assertNotIn(">A<", doc)
        kept = sp.render(doc, "| 2026-11-25 | `c` | gate | 1 | 1 | 1 | 1 | - | - | - |")
        self.assertIn("<!-- load:begin -->B<!-- load:end -->", kept)
        self.assertNotIn("load:begin", sp.render("", "| 2026-11-25 | `c` | gate | 1 | 1 | 1 | 1 | - | - | - |"))

    def test_record_and_main(self):
        d = tempfile.mkdtemp()
        subprocess.run(["git", "init", "-q", d], check=True)
        self.assertEqual(lp.record("nothing", "debug", d), 1)
        self.assertEqual(lp.record(LINE, "debug", d), 0)
        self.assertIn("| json-server |", open(f"{d}/system-profile.md").read())
        self.assertEqual(lp.main([]), 2)
        self.assertEqual(lp.main(["run", "--bogus"]), 2)
        saved = f"{d}/out.txt"
        open(saved, "w").write("no figures\n")
        self.assertEqual(lp.main(["render", saved]), 1)


if __name__ == "__main__":
    unittest.main()
