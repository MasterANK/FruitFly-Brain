from pathlib import Path
import numpy as np

from network import MaleCNSNetwork

# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# ============================================================
# Parameters to test
# ============================================================

GAINS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0]
DTS = [0.01, 0.02, 0.05]
TAUS = [0.5, 1.0, 2.0]

STEPS = 100
INPUT_STRENGTH = 0.2

# ============================================================
# Network
# ============================================================

network = MaleCNSNetwork(
    neuron_file=DATA / "male_cns_navigation_neurons.csv",
    connection_file=DATA / "male_cns_navigation_connections.csv",
)

# ============================================================
# Run one condition
# ============================================================

def run_condition(gain, dt, tau, side):
    network.gain = gain
    network.dt = dt
    network.tau = tau

    network.reset()

    # Input
    if side == "L":
        network.set_side_input("EPG", "L", INPUT_STRENGTH)
    elif side == "R":
        network.set_side_input("EPG", "R", INPUT_STRENGTH)
    elif side == "C":
        network.set_type_input("EPG", INPUT_STRENGTH)

    # Simulate
    for _ in range(STEPS):
        network.step()
        if not np.all(np.isfinite(network.r)):
            return None

    # Network statistics
    activity = network.r
    mean_activity = float(np.mean(activity))
    saturation = float(np.mean(activity > 0.95))

    # PFL populations
    left_values = []
    right_values = []

    for neuron_type in ["PFL1", "PFL2", "PFL3"]:
        left = network.get_side_mean_activity(neuron_type, "L")
        right = network.get_side_mean_activity(neuron_type, "R")
        left_values.append(left)
        right_values.append(right)

    pfl_left = float(np.mean(left_values))
    pfl_right = float(np.mean(right_values))
    difference = pfl_left - pfl_right

    return {
        "mean_activity": mean_activity,
        "saturation": saturation,
        "pfl_left": pfl_left,
        "pfl_right": pfl_right,
        "difference": difference,
        "absolute_difference": abs(difference),
    }

# ============================================================
# Evaluate one parameter configuration
# ============================================================

def evaluate(gain, dt, tau):
    left = run_condition(gain, dt, tau, "L")
    center = run_condition(gain, dt, tau, "C")
    right = run_condition(gain, dt, tau, "R")

    # Numerical failure
    if left is None or center is None or right is None:
        return None

    # 1. Stability / saturation
    avg_saturation = np.mean([left["saturation"], center["saturation"], right["saturation"]])
    saturation_score = 1.0 - avg_saturation

    # 2. Useful overall activity (Ideal activity roughly around 0.5)
    avg_activity = np.mean([left["mean_activity"], center["mean_activity"], right["mean_activity"]])
    activity_score = max(0.0, 1.0 - abs(avg_activity - 0.5) / 0.5)

    # 3. Directional discrimination
    # LEFT input -> positive difference (PFL-L > PFL-R)
    # RIGHT input -> negative difference (PFL-R > PFL-L)
    left_direction = left["difference"]
    right_direction = right["difference"]

    left_correct = max(0.0, left_direction)
    right_correct = max(0.0, -right_direction)
    directional_score = (left_correct + right_correct) / 2.0

    # 4. Center symmetry
    center_difference = abs(center["difference"])
    center_score = max(0.0, 1.0 - center_difference)

    # 5. Direction reversal bonus
    reversal_bonus = 1.0 if (left_direction > 0 and right_direction < 0) else 0.0

    # Final score
    score = (
        0.25 * saturation_score
        + 0.15 * activity_score
        + 0.35 * directional_score
        + 0.10 * center_score
        + 0.15 * reversal_bonus
    )

    return {
        "gain": gain,
        "dt": dt,
        "tau": tau,
        "score": float(score),
        "activity": float(avg_activity),
        "saturation": float(avg_saturation),
        "left_pfl_l": float(left["pfl_left"]),
        "left_pfl_r": float(left["pfl_right"]),
        "right_pfl_l": float(right["pfl_left"]),
        "right_pfl_r": float(right["pfl_right"]),
        "center_pfl_l": float(center["pfl_left"]),
        "center_pfl_r": float(center["pfl_right"]),
        "left_difference": float(left_direction),
        "right_difference": float(right_direction),
        "reversal": reversal_bonus == 1.0,
    }

# ============================================================
# Brute-force search
# ============================================================

print("\n==============================================")
print(" MaleCNS DIRECTIONAL NETWORK TUNING")
print("==============================================\n")

total = len(GAINS) * len(DTS) * len(TAUS)
print(f"Testing {total} configurations\n")

results = []
current = 0

for gain in GAINS:
    for dt in DTS:
        for tau in TAUS:
            current += 1
            print(f"[{current:3d}/{total}] gain={gain:<4} dt={dt:<4} tau={tau:<3}", end=" ")

            result = evaluate(gain, dt, tau)

            if result is None:
                print("FAILED")
            else:
                results.append(result)
                print(f"score={result['score']:.3f}")

# ============================================================
# Sort
# ============================================================

results.sort(key=lambda x: x["score"], reverse=True)

# ============================================================
# Display results
# ============================================================

print("\n==============================================")
print(" BEST CONFIGURATIONS")
print("==============================================\n")
print("Rank | Gain | dt   | tau | Score | Activity | Sat | L→L-R | R→L-R | Rev")
print("-" * 86)

for i, r in enumerate(results[:10], start=1):
    rev_str = "YES" if r["reversal"] else "NO"
    print(
        f"{i:4d} | "
        f"{r['gain']:<4} | "
        f"{r['dt']:<4} | "
        f"{r['tau']:<3} | "
        f"{r['score']:.3f} | "
        f"{r['activity']:.3f} | "
        f"{r['saturation']:.3f} | "
        f"{r['left_difference']:+.3f} | "
        f"{r['right_difference']:+.3f} | "
        f"{rev_str}"
    )

# ============================================================
# Detailed best result
# ============================================================

if results:
    best = results[0]

    print("\n==============================================")
    print(" RECOMMENDED PARAMETERS")
    print("==============================================\n")
    print(f"gain = {best['gain']}")
    print(f"dt   = {best['dt']}")
    print(f"tau  = {best['tau']}\n")

    print(f"Score       = {best['score']:.4f}")
    print(f"Activity    = {best['activity']:.4f}")
    print(f"Saturation  = {best['saturation']:.4f}\n")

    print("LEFT INPUT")
    print(f"  PFL-L = {best['left_pfl_l']:.4f}")
    print(f"  PFL-R = {best['left_pfl_r']:.4f}")
    print(f"  Difference = {best['left_difference']:+.4f}\n")

    print("RIGHT INPUT")
    print(f"  PFL-L = {best['right_pfl_l']:.4f}")
    print(f"  PFL-R = {best['right_pfl_r']:.4f}")
    print(f"  Difference = {best['right_difference']:+.4f}\n")

    print("CENTER INPUT")
    print(f"  PFL-L = {best['center_pfl_l']:.4f}")
    print(f"  PFL-R = {best['center_pfl_r']:.4f}\n")

    print("Direction reversal =", "YES" if best["reversal"] else "NO")
else:
    print("\nNo stable configurations found.")