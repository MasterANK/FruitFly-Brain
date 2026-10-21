from pathlib import Path
import numpy as np

from network import MaleCNSNetwork

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

network = MaleCNSNetwork(
    neuron_file=DATA / "male_cns_navigation_neurons.csv",
    connection_file=DATA / "male_cns_navigation_connections.csv",
    gain=0.75,
    dt=0.01,
    tau=0.5,
)


def run_condition(side):
    network.reset()

    if side == "L":
        network.set_side_input("EPG", "L", 0.2)
    elif side == "R":
        network.set_side_input("EPG", "R", 0.2)
    elif side == "C":
        network.set_type_input("EPG", 0.2)

    for _ in range(100):
        network.step()

    left = []
    right = []

    for neuron_type in ["PFL1", "PFL2", "PFL3"]:
        left.append(network.get_side_mean_activity(neuron_type, "L"))
        right.append(network.get_side_mean_activity(neuron_type, "R"))

    pfl_left = float(np.mean(left))
    pfl_right = float(np.mean(right))

    return pfl_left, pfl_right


print("\n========================================")
print("CONTROLLER VERIFICATION")
print("========================================")

left_l, left_r = run_condition("L")
right_l, right_r = run_condition("R")
center_l, center_r = run_condition("C")

print("\nLEFT INPUT")
print(f"PFL-L = {left_l:.6f}")
print(f"PFL-R = {left_r:.6f}")
print(f"Difference = {left_l - left_r:+.6f}")

print("\nRIGHT INPUT")
print(f"PFL-L = {right_l:.6f}")
print(f"PFL-R = {right_r:.6f}")
print(f"Difference = {right_l - right_r:+.6f}")

print("\nCENTER INPUT")
print(f"PFL-L = {center_l:.6f}")
print(f"PFL-R = {center_r:.6f}")
print(f"Difference = {center_l - center_r:+.6f}")

left_correct = left_l > left_r
right_correct = right_r > right_l

print("\n========================================")
print("RESULT")
print("========================================")
print("LEFT  → PFL-L > PFL-R:", left_correct)
print("RIGHT → PFL-R > PFL-L:", right_correct)
print("Direction reversal:", left_correct and right_correct)