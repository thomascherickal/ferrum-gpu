"""Part of system_profile.py: reading the machine. /proc on Linux; elsewhere the load average (Unix)
or nothing (Windows). A reading that is not available is None and shows as "-"; nothing is estimated."""
import os
import subprocess


def cpu_busy(a, b):
    """CPU busy % between two `/proc/stat` first lines (user nice system idle iowait ...)."""
    x = [int(v) for v in a.split()[1:]]
    y = [int(v) for v in b.split()[1:]]
    idle = (y[3] + y[4]) - (x[3] + x[4])
    total = sum(y) - sum(x)
    return 100.0 * (total - idle) / total if total > 0 else 0.0


def mem_used_mib(meminfo):
    v = {k: int(rest.split()[0]) for k, rest in (line.split(":", 1) for line in meminfo.splitlines() if ":" in line)}
    return (v["MemTotal"] - v["MemAvailable"]) / 1024.0


def gpu_reading(text):
    """(busy %, memory MiB) of the busiest GPU in `nvidia-smi --query-gpu=utilization.gpu,memory.used`."""
    best = None
    for line in text.strip().splitlines():
        try:
            u, m = (float(x) for x in line.replace("%", "").replace("MiB", "").split(","))
        except ValueError:
            continue
        best = (u, m) if best is None or u > best[0] else best
    return best


def read_gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return gpu_reading(out)


def stat_line(path="/proc/stat"):
    """The first line of /proc/stat, or None where there is no /proc (macOS, Windows)."""
    try:
        with open(path) as f:
            return f.readline()
    except OSError:
        return None


def load_busy(getloadavg=getattr(os, "getloadavg", None), cpus=None):
    """CPU busy % estimated from the 1-minute load average over the CPU count; None if unavailable."""
    try:
        return min(100.0, 100.0 * getloadavg()[0] / (cpus or os.cpu_count() or 1))
    except (TypeError, OSError):
        return None


def cpu_now(prev, cur, load=load_busy):
    """Busy % between two /proc/stat lines, else the load-average estimate."""
    return cpu_busy(prev, cur) if prev and cur else load()


def mem_now(path="/proc/meminfo"):
    """Memory in use (MiB) from /proc/meminfo, or None where it does not exist."""
    try:
        with open(path) as f:
            return mem_used_mib(f.read())
    except (OSError, KeyError):
        return None


def blank(v, digits):
    return "" if v is None else f"{v:.{digits}f}"
