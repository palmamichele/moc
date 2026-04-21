import re
import pandas as pd
import numpy as np 
from pathlib import Path

def parse_value(x):
    """Convert plain floats and strings of numbers to float."""
    if isinstance(x, (int, float)):
        return float(x)
    x = str(x).strip()

    # Try direct float first
    try:
        return float(x)
    except ValueError:
        pass

    # Extract the first numeric token
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', x)
    if not m:
        raise ValueError(f"Could not parse numeric value from: {x}")
    return float(m.group())

max_diff = float("-inf")
max_file = None
diffs = {}


for k in range(100):
    path= Path("experiments")/"linear-california"/f"model_{k}.csv"
    df = pd.read_csv(str(path))

    print(df)

    trivial_val = parse_value(
        df.loc[df["constant type"] == "trivial", "value"].iloc[0]
    )
    dmoc_union_val = parse_value(
        df.loc[df["constant type"] == "dmoc_union", "value"].iloc[0]
    )

    diff = np.abs(dmoc_union_val - trivial_val)
    diffs[k] = diff

    if diff > max_diff:
        max_diff = diff
        max_file = k

print("Per-file diffs:")
for path, diff in diffs.items():
    print(f"{path}: {diff}")

print(f"\nMax diff: {max_diff}")
print(f"File with max diff: {max_file}")