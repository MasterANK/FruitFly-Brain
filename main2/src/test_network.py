from pathlib import Path

import numpy as np

from network import MaleCNSNetwork


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


network = MaleCNSNetwork(
    neuron_file=DATA / "male_cns_navigation_neurons.csv",
    connection_file=DATA / "male_cns_navigation_connections.csv",
)


print("\nAvailable neuron types:")
print(network.available_types())


# --------------------------------------------------
# Give one neuron a small external stimulus
# --------------------------------------------------

network.set_type_input(
    "EPG",
    0.5
)

print("\nRunning network...")

for step in range(100):

    activity = network.step()

    if step % 10 == 0:

        print(
            f"Step {step:3d} | "
            f"min={activity.min():.4f} | "
            f"max={activity.max():.4f} | "
            f"mean={activity.mean():.4f}"
        )

print("\nNavigation circuit activity:")

for neuron_type in [
    "EPG",
    "Delta7",
    "PFL1",
    "PFL2",
    "PFL3",
]:

    print(
        f"{neuron_type:8s}: "
        f"{network.get_type_mean_activity(neuron_type):.6f}"
    )

print("\nPFL diagnostic:")

for neuron_type in ["PFL1", "PFL2", "PFL3"]:

    indices = network.type_to_indices.get(
        neuron_type,
        []
    )

    print(f"\n{neuron_type}")

    for index in indices[:5]:

        recurrent = network.W[index] @ network.r

        print(
            f"neuron {index}: "
            f"activity={network.r[index]:.6f}, "
            f"recurrent_input={recurrent:.6f}"
        )
# --------------------------------------------------
# Final checks
# --------------------------------------------------

print("\n========================================")
print("SANITY CHECK")
print("========================================")

print(
    "NaN:",
    np.isnan(activity).any()
)

print(
    "Infinite:",
    np.isinf(activity).any()
)

print(
    "Active neurons:",
    np.count_nonzero(activity > 0)
)

print(
    "Maximum activity:",
    activity.max()
)

print(
    "Mean activity:",
    activity.mean()
)