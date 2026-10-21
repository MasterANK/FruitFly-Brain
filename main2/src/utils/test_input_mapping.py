"""
USED to check is EPG tuning could help in l/r diff
"""

from pathlib import Path
from network import MaleCNSNetwork

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

network = MaleCNSNetwork(
    neuron_file=DATA / "male_cns_navigation_neurons.csv",
    connection_file=DATA / "male_cns_navigation_connections.csv",
    gain=0.75,
    dt=0.01,
    tau=0.5
)


def run_input(side, strength):
    network.reset()

    network.set_side_input("EPG", side, strength)

    for _ in range(200):
        network.step()

    pfl1_l = network.get_side_mean_activity("PFL1", "L")
    pfl1_r = network.get_side_mean_activity("PFL1", "R")

    pfl2_l = network.get_side_mean_activity("PFL2", "L")
    pfl2_r = network.get_side_mean_activity("PFL2", "R")

    pfl3_l = network.get_side_mean_activity("PFL3", "L")
    pfl3_r = network.get_side_mean_activity("PFL3", "R")

    pfl_l = (pfl1_l + pfl2_l + pfl3_l) / 3
    pfl_r = (pfl1_r + pfl2_r + pfl3_r) / 3

    return pfl_l, pfl_r

strengths = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

print("\n========================================")
print("EPG INPUT STRENGTH TEST")
print("========================================")

for strength in strengths:

    left_l, left_r = run_input("L", strength)
    right_l, right_r = run_input("R", strength)

    left_diff = left_l - left_r
    right_diff = right_l - right_r

    reversal = left_diff > 0 and right_diff < 0

    separation = abs(left_diff) + abs(right_diff)

    print(
        f"{strength:.2f} | "
        f"L: {left_diff:+.4f} | "
        f"R: {right_diff:+.4f} | "
        f"Sep: {separation:.4f} | "
        f"Reversal: {reversal}"
    )