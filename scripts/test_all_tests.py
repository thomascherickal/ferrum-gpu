"""Tests for all_tests.py and all_tests_reports.py: python3 scripts/test_all_tests.py"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import all_tests as a  # noqa: E402
import all_tests_reports as r  # noqa: E402

RUN = """Running unittests src/lib.rs (target/debug/deps/ferrum_tok-0123456789abcdef)
test x ... ok
test bad::one ... FAILED
test result: FAILED. 3 passed; 1 failed; 2 ignored; 0 measured
   Doc-tests ferrum_tok
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured
Running tests/api.rs (target/debug/deps/api-0123456789abcdef)
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured
"""


def totals(lines, files):
    cell = lambda p, c, n: {"percent": p, "covered": c, "count": n}  # noqa: E731
    file = lambda name: {"filename": name, "summary": {"lines": cell(50.0, 1, 2), "regions": cell(50.0, 1, 2),
                                                       "functions": cell(100.0, 1, 1)}}
    return {"totals": {"lines": lines, "regions": cell(99.5, 199, 200), "functions": cell(100.0, 9, 9)},
            "files": [file(f) for f in files]}


class Reports(unittest.TestCase):
    def test_test_output_is_split_per_binary_with_its_failures(self):
        rows, failed = r.parse(RUN.splitlines())
        self.assertEqual(rows, [("ferrum_tok", 3, 1, 2), ("doc:ferrum_tok", 4, 0, 0), ("api", 0, 0, 0)])
        self.assertEqual(failed, ["bad::one"])

    def test_a_section_sums_the_binaries_and_lists_the_failures(self):
        md, passed, failed, ignored = r.test_section("With the GPU", *r.parse(RUN.splitlines()))
        self.assertEqual((passed, failed, ignored), (7, 1, 2))
        self.assertEqual(md[0], "## With the GPU: 7 passed, 1 failed, 2 ignored\n")
        self.assertIn("| ferrum_tok | 3 | 1 | 2 |", md)
        self.assertNotIn("| api | 0 | 0 | 0 |", md)
        self.assertEqual(md[-3:], ["Failed tests:", "- `bad::one`", ""])

    def test_the_coverage_report_names_the_total_and_every_file_relative_to_the_root(self):
        lines = {"percent": 99.0, "covered": 99, "count": 100}
        text, t = r.coverage_report(totals(lines, ["/r/b.rs", "/r/a.rs"]), "/r/", "now", "abc")
        self.assertIn("**Total: 99.00% of lines (1 of 100 missed), 99.50% of regions, 100.00% of functions.**", text)
        self.assertLess(text.index("| a.rs |"), text.index("| b.rs |"))
        self.assertEqual(t["lines"]["percent"], 99.0)
        self.assertEqual(r.lowest(text, 1), [" a.rs  50.00% "])

    def test_the_noise_filter_and_the_lowest_files(self):
        self.assertEqual(a.NOT_EGL, "libEGL")
        low = "| file | lines |\n|---|---:|\n| a | 90.00% |\n| b | 20.00% |\n"
        self.assertEqual([" ".join(x.split()) for x in r.lowest(low, 5)], ["b 20.00%", "a 90.00%"])
        self.assertEqual(len(r.lowest(low, 1)), 1)


if __name__ == "__main__":
    unittest.main()
