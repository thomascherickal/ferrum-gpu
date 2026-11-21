"""Tests for metrics: python3 scripts/test_metrics_render.py"""
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as m  # noqa: E402
import metrics_collect as mc  # noqa: E402

SUMMARY = """Filename            Regions  Missed Regions  Cover  Functions  Missed Functions  Executed  Lines  Missed Lines  Cover  Branches  Missed Branches  Cover
-------------------------------------------------------------------------------------------------------------------------------------------
ferrum_a/src/x.rs        10               1  90.00%          4                 0   100.00%     40             4  90.00%         0                0      -
ferrum_a/src/y.rs        20               0 100.00%          2                 0   100.00%     60             0 100.00%         0                0      -
ferrum_b/src/z.rs        30               6  80.00%          6                 2    66.67%    100            10  90.00%         0                0      -
-------------------------------------------------------------------------------------------------------------------------------------------
TOTAL                    60               7  88.33%         12                 2    83.33%    200            14  93.00%         0                0      -
"""




class Render(unittest.TestCase):
    def test_summary_parsing(self):
        rows, total = m.parse_summary(SUMMARY)
        self.assertEqual([r["name"] for r in rows], ["ferrum_a/src/x.rs", "ferrum_a/src/y.rs", "ferrum_b/src/z.rs"])
        self.assertEqual((total["lines"], total["missed_lines"], total["regions"]), (200, 14, 60))
        self.assertEqual(m.parse_summary("nothing\nuseful"), ([], None))
        self.assertEqual(m.parse_summary("a.rs x y z 1 2 3 4 5 6")[0], [])

    def test_coverage_block(self):
        b = m.coverage_block(SUMMARY, "98", "abc1234", "2026-11-18 10:00 UTC")
        self.assertIn("| lines | 186 | 200 | 14 | 93.00% |", b)
        self.assertIn("| ferrum_a | 100 | 4 | 96.00% | 96.67% |", b)
        self.assertIn("| ferrum_b | 100 | 10 | 90.00% | 80.00% |", b)
        self.assertIn("**98%**", b)
        self.assertIn("`abc1234`", b)
        # the file with the most missed lines is listed first, fully covered files are not listed
        self.assertLess(b.index("ferrum_b/src/z.rs"), b.index("ferrum_a/src/x.rs"))
        self.assertNotIn("y.rs", b)
        self.assertEqual(m.coverage_block("garbage", "98", "x", "now"), m.NONE_YET)
        absolute = SUMMARY.replace("ferrum_a/", "/repo/ferrum_a/").replace("ferrum_b/", "/repo/ferrum_b/")
        self.assertIn("| ferrum_a | 100 | 4 |", m.coverage_block(absolute, "98", "c", "t", "/repo"))
        full = SUMMARY.replace("            4  90.00%", "            0 100.00%")
        self.assertIsNotNone(m.coverage_block(full.replace("ferrum_b/src/z.rs", "ferrum_b/src/zz.rs"), "100", "c", "t"))

    def test_gate_block(self):
        text = "start rc=0 10:00:00\nfmt rc=0 10:00:30\nxe rc=1 10:05:30\ncoverage rc=0 00:10:30\n"
        b = m.gate_block(text, "FAIL\n")
        self.assertIn("| fmt | pass | 0m 30s |", b)
        self.assertIn("| xe | FAIL (1) | 5m 00s |", b)
        self.assertIn("| coverage | pass | ", b)
        self.assertIn("| **total** | FAIL | ", b)
        self.assertEqual(m.gate_block("start rc=0 10:00:00\n", "PASS"), m.NONE_YET)
        self.assertEqual(m.seconds("01:02:03"), 3723)

    def test_blocks_wrap_and_unwrap(self):
        w = m.wrap(m.GATE_MARK, "body")
        self.assertEqual(m.get_block("x\n" + w + "\ny", m.GATE_MARK), "body")
        self.assertIsNone(m.get_block("no markers", m.GATE_MARK))
        self.assertEqual(m.pct(1, 4), "25.00%")
        self.assertEqual(m.pct(0, 0), "-")


if __name__ == "__main__":
    unittest.main()
