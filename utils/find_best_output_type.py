"""
find_best_output_type.py

ER1_a's direct anatomical path to PFL1 is very thin (1 synapse), while
Delta7's non-topographic inhibition dominates PFL1 by a wide margin
(259 synapses). That structural imbalance likely explains why genuine
(correctly hemisphere-labeled) steering differentiation through PFL1 is
weak. This script tests several OTHER candidate output types that sit
structurally closer to EPG -- PFN variants -- to see whether any of
them carry a stronger genuine left/right signal.

Critically, this ONLY trusts results where BOTH the input and output
types parsed with real (non-fallback) hemisphere splits -- any
combination that silently fell back to an arbitrary index split is
excluded from consideration, so we don't repeat the earlier mistake of
trusting a spuriously good-looking but meaningless number.

Run from environment/ folder.
"""

import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "main"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main.config import build_network, INPUT_TYPE
from connectome_agent import ConnectomeAgent, hemisphere_of
from world import SensorReading

CANDIDATE_OUTPUT_TYPES = ["PFL1", "PFL2", "PFL3", "PFNd", "PFNv",
                          "PFNp_a", "PFNp_b", "PFNp_c", "PFNp_d", "PFNp_e",
                          "PFNm_a", "PFNm_b"]

INH_GAIN_VALUES = [4.0, 6.0, 8.0, 10.0]
N_SUBSTEPS = 40
SENSOR_GAIN = 8.0


def hemisphere_split_is_real(net, cell_type):
    """Checks whether a cell type has genuine, parseable hemisphere
    labels (not a fallback situation) BEFORE we bother testing it."""
    rows = net.neurons[net.neurons["type"] == cell_type]
    if len(rows) == 0:
        return False, 0, 0
    left = sum(1 for _, r in rows.iterrows() if hemisphere_of(r["instance"]) == "L")
    right = sum(1 for _, r in rows.iterrows() if hemisphere_of(r["instance"]) == "R")
    return (left > 0 and right > 0), left, right


def evaluate(output_type, inh_gain):
    net = build_network()
    net.inh_gain = inh_gain

    is_real, left_n, right_n = hemisphere_split_is_real(net, output_type)
    if not is_real:
        return None  # skip -- would silently use a meaningless fallback split

    agent = ConnectomeAgent(net, input_type=INPUT_TYPE, output_type=output_type,
                             sensor_gain=SENSOR_GAIN, turn_gain=1.0, n_substeps=N_SUBSTEPS)

    agent.reset()
    for _ in range(100):
        agent.net.step()
    reading_left = SensorReading(light_angle=None, light_distance=None,
                                  food_angle=math.radians(60), food_distance=100, collided=False)
    turn_left, _ = agent.step(reading_left)

    agent.reset()
    for _ in range(100):
        agent.net.step()
    reading_right = SensorReading(light_angle=None, light_distance=None,
                                   food_angle=math.radians(-60), food_distance=100, collided=False)
    turn_right, _ = agent.step(reading_right)

    has_nan = np.any(np.isnan(agent.net.r))
    diff = abs(turn_left - turn_right) if not has_nan else -1.0
    return diff, turn_left, turn_right, left_n, right_n, has_nan


def main():
    print(f"Testing {len(CANDIDATE_OUTPUT_TYPES)} candidate output types x "
          f"{len(INH_GAIN_VALUES)} inhibition strengths for genuine "
          f"(non-fallback) left/right steering signal...\n")

    # Sanity check: confirm INPUT_TYPE itself has real hemisphere labels
    # before testing anything else -- this was silently broken before a
    # recent fix (ER1_a's naming has no digit after the hemisphere
    # letter, e.g. 'ER1_a(ring)_L', which an earlier regex missed).
    probe_net = build_network()
    input_is_real, input_left, input_right = hemisphere_split_is_real(probe_net, INPUT_TYPE)
    if not input_is_real:
        print(f"[!] STOP: INPUT_TYPE='{INPUT_TYPE}' does not have real hemisphere "
              f"labels ({input_left} left, {input_right} right found). All "
              f"results below would be meaningless (input side would silently "
              f"fall back to an arbitrary split). Fix hemisphere_of() for this "
              f"type's naming convention before continuing.")
        return
    else:
        print(f"Confirmed INPUT_TYPE='{INPUT_TYPE}' has real hemisphere labels "
              f"({input_left} left, {input_right} right). Proceeding.\n")

    results = []
    for output_type in CANDIDATE_OUTPUT_TYPES:
        for inh_gain in INH_GAIN_VALUES:
            r = evaluate(output_type, inh_gain)
            if r is None:
                print(f"  {output_type:<10} inh_gain={inh_gain:<5} -> SKIPPED "
                      f"(no real hemisphere labels found for this type)")
                continue
            diff, tl, tr, left_n, right_n, has_nan = r
            print(f"  {output_type:<10} inh_gain={inh_gain:<5} -> diff={diff:.5f}  "
                  f"(turn_left={tl:+.5f}, turn_right={tr:+.5f}, "
                  f"{left_n}L/{right_n}R neurons)"
                  f"{'  [NaN]' if has_nan else ''}")
            results.append((diff, output_type, inh_gain, tl, tr))

    if not results:
        print("\n[!] No candidate output type had genuine hemisphere labels at all. "
              "Report this back -- we'd need to reconsider the whole steering "
              "approach, e.g. driving turning from EPG's own hemisphere split "
              "directly instead of any downstream output type.")
        return

    results.sort(key=lambda r: r[0], reverse=True)
    print("\nTop 5 genuine (non-fallback) results:")
    for diff, output_type, inh_gain, tl, tr in results[:5]:
        print(f"  {output_type:<10} inh_gain={inh_gain:<5} diff={diff:.5f}  "
              f"(turn_left={tl:+.5f}, turn_right={tr:+.5f})")

    best_diff, best_type, best_inh_gain, tl, tr = results[0]
    print(f"\nBest candidate: output_type='{best_type}', inh_gain={best_inh_gain}, "
          f"difference={best_diff:.5f}")
    if best_diff < 0.01:
        print("\n[!] Even the best candidate is weak. Next step would be driving "
              "input directly into EPG's own hemisphere split and reading "
              "steering from EPG or PEN directly, rather than routing through "
              "any single downstream output type.")
    else:
        print(f"\nThis is meaningfully better than PFL1's ~0.008. Consider "
              f"switching OUTPUT_TYPE to '{best_type}' in config.py and "
              f"re-running auto_tune_steering.py with this as the new output_type.")


if __name__ == "__main__":
    main()