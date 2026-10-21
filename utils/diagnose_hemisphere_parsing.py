"""
diagnose_hemisphere_parsing.py

A single, clean, deterministic check of hemisphere parsing for ER1_a and
PFL1 -- printing every instance string and what hemisphere_of() extracts
from it. This bypasses any ambiguity from reading scrolled/interleaved
output of a long grid search, since parsing a fixed CSV is inherently
deterministic and should give the exact same result every time.

Run from environment/ folder.
"""

import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from connectome_agent import hemisphere_of
from main.config import NEURONS_CSV, INPUT_TYPE, OUTPUT_TYPE


def check_type(df, cell_type):
    rows = df[df["type"] == cell_type]
    print(f"\n--- {cell_type} ({len(rows)} neurons) ---")
    left, right, unlabeled = 0, 0, 0
    for _, row in rows.iterrows():
        side = hemisphere_of(row["instance"])
        print(f"  neuron_id={row['neuron_id']:<12} instance={row['instance']:<30} -> {side}")
        if side == "L":
            left += 1
        elif side == "R":
            right += 1
        else:
            unlabeled += 1
    print(f"  TOTAL: {left} left, {right} right, {unlabeled} unlabeled")
    return left, right, unlabeled


def main():
    df = pd.read_csv(NEURONS_CSV)
    print(f"Loaded {len(df)} neurons from {NEURONS_CSV}")

    check_type(df, INPUT_TYPE)
    check_type(df, OUTPUT_TYPE)


if __name__ == "__main__":
    main()