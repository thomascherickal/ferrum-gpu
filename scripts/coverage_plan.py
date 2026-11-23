"""The commands and records of coverage.py, apart from running them."""
import hashlib
import sys

PY = sys.executable
TIMING = "target/coverage-timing.txt"
SUMMARY = "target/coverage-summary.txt"
LCOV = "target/coverage.lcov"
LAYOUT = "target/coverage-layout"
NO_INTEROP = "not binary(interop)"
ADAPTER_RUN = ("package(ferrum_wgpu) | package(ferrum_gpu_core) | (package(ferrum_gpu_gui) & "
               "(kind(lib) | kind(bin) | binary(api) | binary(binary)))")


def layout_hash(files):
    """sha1 of the sorted source-file list, one per line: it changes when a file is added, moved or removed."""
    return hashlib.sha1(("\n".join(sorted(files)) + "\n").encode()).hexdigest()


def timing_line(name, seconds):
    return f"{name} {int(seconds)}s"


def run_one():
    """Everything on the default adapter, except the interoperability tests (real llama.cpp and Ollama)."""
    return ["cargo", "llvm-cov", "nextest", "--workspace", "--no-report", "-E", NO_INTEROP]


def run_two():
    """No adapter, for the "skip when there is no adapter" branches: only the crates that open one."""
    return ["cargo", "llvm-cov", "nextest", "--no-report", "-p", "ferrum_wgpu", "-p", "ferrum_gpu_core",
            "-p", "ferrum_gpu_gui", "-E", ADAPTER_RUN]


def summary_cmd(floor):
    """The llvm-cov summary; with a floor it also fails when total line coverage is below it."""
    cmd = ["cargo", "llvm-cov", "report", "--summary-only"]
    return cmd + ["--fail-under-lines", floor] if floor else cmd


def lcov_cmd():
    return ["cargo", "llvm-cov", "report", "--lcov", "--output-path", LCOV]


def metrics_cmd(floor):
    return [PY, "scripts/metrics.py", "--coverage", SUMMARY, "--floor", floor or "none"]
