"""Tests for system_profile.py: python3 scripts/test_system_profile.py"""
import os
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import system_profile as p  # noqa: E402


class Profile(unittest.TestCase):
    def test_cpu_memory_and_gpu_readings(self):
        self.assertAlmostEqual(p.cpu_busy("cpu 10 0 10 80 0", "cpu 40 0 40 120 0"), 100 * 60 / 100)
        self.assertEqual(p.cpu_busy("cpu 1 0 1 1 0", "cpu 1 0 1 1 0"), 0.0)
        self.assertAlmostEqual(p.mem_used_mib("MemTotal: 2048 kB\nMemAvailable: 1024 kB\nX: 1 kB"), 1.0)
        self.assertEqual(p.gpu_reading("5, 100\n80, 900\n"), (80.0, 900.0))
        self.assertIsNone(p.gpu_reading("garbage\n"))
        self.assertIsNone(p.gpu_reading(""))

    def test_summary_and_rows(self):
        s = p.summarize("2,10.0,1000,,\n4,30.0,2000,50,300\n6,20.0,1500,70,400\n")
        self.assertEqual((s["seconds"], s["cpu_avg"], s["cpu_peak"], s["mem_peak"]), (6.0, 20.0, 30.0, 2000.0))
        self.assertEqual((s["gpu_avg"], s["gpu_peak"], s["gpu_mem_peak"]), (60.0, 70.0, 400.0))
        no_gpu = p.summarize("2,10.0,1000,,\n")
        self.assertIsNone(no_gpu["gpu_peak"])
        self.assertIn("| - | - | - |", p.row(no_gpu, "abc", "gate", "2026-11-21 10:00")[-20:] + " | - | - | - |")
        self.assertIsNone(p.summarize(""))
        self.assertTrue(p.row(s, "abc", "gate", "2026-11-21 10:00").startswith("| 2026-11-21 10:00 | `abc` | gate | 6 |"))

    def test_the_history_keeps_the_newest_rows_first_and_a_bounded_number(self):
        doc = ""
        for i in range(p.KEEP + 3):
            doc = p.render(doc, f"| 2026-11-{i:02d} | `c{i}` | gate | 1 | 1 | 1 | 1 | - | - | - |")
        rows = [l for l in doc.splitlines() if l.startswith("| 20")]
        self.assertEqual(len(rows), p.KEEP)
        self.assertIn("c%d" % (p.KEEP + 2), rows[0])
        self.assertIn(p.BEGIN, doc)

    def test_a_machine_without_proc_falls_back_to_load_and_leaves_memory_empty(self):
        self.assertIsNone(p.stat_line("/no/such/stat"))
        self.assertIsNone(p.mem_now("/no/such/meminfo"))
        d = tempfile.mkdtemp()
        bad = f"{d}/meminfo"
        with open(bad, "w") as f:
            f.write("nothing useful\n")
        self.assertIsNone(p.mem_now(bad))
        self.assertEqual(p.stat_line(bad), "nothing useful\n")
        self.assertEqual(p.load_busy(lambda: (2.0, 0, 0), cpus=4), 50.0)
        self.assertEqual(p.load_busy(lambda: (9.0, 0, 0), cpus=4), 100.0)
        self.assertIsNone(p.load_busy(None))
        self.assertIsNone(p.load_busy(lambda: (_ for _ in ()).throw(OSError())))
        self.assertEqual(p.cpu_now("cpu 10 0 10 80 0", "cpu 40 0 40 120 0"), 60.0)
        self.assertEqual(p.cpu_now(None, None, load=lambda: 7.5), 7.5)
        self.assertEqual((p.blank(None, 1), p.blank(2.26, 1)), ("", "2.3"))
        only_load = p.summarize("2,,,,\n4,,,,\n")
        self.assertIsNone(only_load["cpu_peak"])
        self.assertIsNone(only_load["mem_peak"])
        self.assertIn("| - | - | - |", p.row(only_load, "abc", "x", "t"))

    def test_sampling_stops_when_asked_and_main_handles_its_arguments(self):
        d = tempfile.mkdtemp()
        f = f"{d}/s.csv"
        threading.Thread(target=lambda: (time.sleep(0.3), open(f + ".stop", "w").close())).start()
        p.sample(f, interval=0.05)
        self.assertTrue(os.path.exists(f))
        self.assertEqual(p.main([]), 2)
        empty = f"{d}/e.csv"
        open(empty, "w").close()
        self.assertEqual(p.main(["render", empty]), 1)


if __name__ == "__main__":
    unittest.main()
