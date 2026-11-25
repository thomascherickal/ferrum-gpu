"""Fixtures the metrics tests share."""

# An llvm-cov summary with two crates, one file under its function and line targets.
SUMMARY = """Filename            Regions  Missed Regions  Cover  Functions  Missed Functions  Executed  Lines  Missed Lines  Cover  Branches  Missed Branches  Cover
-------------------------------------------------------------------------------------------------------------------------------------------
ferrum_a/src/x.rs        10               1  90.00%          4                 0   100.00%     40             4  90.00%         0                0      -
ferrum_a/src/y.rs        20               0 100.00%          2                 0   100.00%     60             0 100.00%         0                0      -
ferrum_b/src/z.rs        30               6  80.00%          6                 2    66.67%    100            10  90.00%         0                0      -
-------------------------------------------------------------------------------------------------------------------------------------------
TOTAL                    60               7  88.33%         12                 2    83.33%    200            14  93.00%         0                0      -
"""
