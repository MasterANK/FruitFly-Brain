"""
verify_config.py

Final sanity check before moving to Phase 4. Builds the network EXACTLY
the way all future code (environment, agent, evaluation) will build it --
via config.build_network() -- rather than through auto_tune.py's internal
search loop. Confirms the winning configuration is genuinely reproducible
from a clean start, not an artifact of shared state during the search.

Run from simulation/ folder (config.py should already exist, written by
auto_tune.py).
"""

import sys
import os
import numpy as np
from pathlib import Path

# Move up 2 levels to the shared parent folder (not into /main)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Keep the 'main.' prefix so VS Code can track it
from main.config import build_network, INPUT_TYPE, OUTPUT_TYPE


def main():
    print("Building network fresh from config.py ...")
    net = build_network()
    print(f"gain={net.gain}, inh_gain={net.inh_gain}, tau={net.tau}, n={net.n}\n")

    # Let the seeded bump settle (mirrors what auto_tune measured)
    for _ in range(150):
        net.step()

    epg = net.get_output_by_type("EPG")
    pfl_before = net.get_output_by_type(OUTPUT_TYPE).copy()

    active_frac = (epg > 0.05).mean()
    print(f"EPG active fraction: {active_frac:.2f}  (want roughly 0.2-0.8, not 0 or 1)")
    print(f"EPG values: {np.round(epg, 2)}\n")
    print(f"{OUTPUT_TYPE} baseline values: {np.round(pfl_before, 3)}")
    print(f"{OUTPUT_TYPE} baseline std: {pfl_before.std():.3f}  (want clearly > 0)\n")

    # Apply sensory input and confirm it shifts things
    for _ in range(100):
        net.set_input_by_type(INPUT_TYPE, 3.0)
        net.step()
    pfl_after = net.get_output_by_type(OUTPUT_TYPE)
    shift = np.abs(pfl_after - pfl_before).mean()

    print(f"{OUTPUT_TYPE} values after {INPUT_TYPE} input: {np.round(pfl_after, 3)}")
    print(f"Mean absolute shift caused by input: {shift:.3f}  (want clearly > 0)\n")

    has_nan = np.any(np.isnan(net.r))
    healthy = (not has_nan) and (0.1 < active_frac < 0.9) and (pfl_before.std() > 0.02) and (shift > 0.02)

    print("=" * 60)
    if healthy:
        print("RESULT: Configuration looks healthy and reproducible.")
        print("Safe to proceed to Phase 4 (the environment) using config.build_network().")
    else:
        print("RESULT: Something looks off when rebuilt fresh -- report this back "
              "before moving on, since it suggests the search result doesn't "
              "reproduce outside auto_tune.py's internal loop.")
    print("=" * 60)


if __name__ == "__main__":
    main()