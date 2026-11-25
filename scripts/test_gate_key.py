"""Tests for gate_key.py: python3 scripts/test_gate_key.py"""
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_key as gk  # noqa: E402


def git(d, *a):
    subprocess.run(["git", "-C", d, "-c", "user.name=t", "-c", "user.email=t@t", *a], check=True, capture_output=True)


def put(d, rel, text):
    os.makedirs(os.path.dirname(f"{d}/{rel}"), exist_ok=True)
    open(f"{d}/{rel}", "w").write(text)


class GateKey(unittest.TestCase):
    def repo(self):
        d = tempfile.mkdtemp()
        git(d, "init", "-q")
        put(d, "a.rs", "fn a() {}\n")
        put(d, "metrics.md", "m1\n")
        git(d, "add", "-A")
        return d

    def record(self, d, verdict, key):
        put(d, ".gate/gate.done", verdict + "\n")
        put(d, ".gate/key", key + "\n")

    def test_a_pass_recorded_for_this_content_is_accepted(self):
        d = self.repo()
        self.assertEqual(gk.check(d)[0], 1)  # nothing recorded
        self.record(d, "PASS", gk.key(d))
        self.assertEqual(gk.check(d)[0], 0)

    def test_generated_reports_may_change_but_code_may_not(self):
        d = self.repo()
        self.record(d, "PASS", gk.key(d))
        put(d, "metrics.md", "m2\n")
        self.assertEqual(gk.check(d)[0], 0)
        put(d, "a.rs", "fn b() {}\n")
        git(d, "add", "-A")
        self.assertEqual(gk.check(d)[0], 1)

    def test_a_failed_gate_and_unstaged_changes_never_pass(self):
        d = self.repo()
        self.record(d, "FAIL", gk.key(d))
        self.assertEqual(gk.check(d)[0], 1)
        self.record(d, "PASS", gk.key(d))
        put(d, "a.rs", "fn changed() {}\n")  # modified after staging
        code, message = gk.check(d)
        self.assertEqual(code, 3)
        self.assertIn("a.rs", message)
        put(d, "new.rs", "x\n")  # untracked
        self.assertIn("new.rs", gk.unstaged(d))

    def test_python_bytecode_caches_are_ignored(self):
        d = self.repo()
        put(d, "scripts/__pycache__/x.cpython-312.pyc", "c1")
        put(d, "y.pyc", "c1")
        git(d, "add", "-A")
        self.record(d, "PASS", gk.key(d))
        put(d, "scripts/__pycache__/x.cpython-312.pyc", "c2")  # rewritten by a run
        put(d, "y.pyc", "c2")
        self.assertEqual(gk.unstaged(d), [])
        git(d, "add", "scripts", "y.pyc")  # even staged, the key is unchanged
        self.assertEqual(gk.check(d)[0], 0)
        self.assertTrue(gk.bytecode("a/__pycache__/b") and not gk.bytecode("a.py"))

    def test_main_prints_the_key(self):
        self.assertEqual(len(gk.key(gk.ROOT)), 40)
        self.assertEqual(gk.main([]), 0)


if __name__ == "__main__":
    unittest.main()
