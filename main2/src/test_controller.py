from pathlib import Path

from network import MaleCNSNetwork


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


network = MaleCNSNetwork(
    neuron_file=DATA / "male_cns_navigation_neurons.csv",
    connection_file=DATA / "male_cns_navigation_connections.csv",
)


print("\nAvailable types:")
print(network.available_types())


# --------------------------------------------------
# Reset
# --------------------------------------------------

network.reset()


# --------------------------------------------------
# Apply an abstract left-side sensory stimulus
# --------------------------------------------------

network.set_side_input(
    "EPG",
    "L",
    0.2,
)


# --------------------------------------------------
# Run
# --------------------------------------------------

for step in range(100):

    network.step()


# --------------------------------------------------
# Read PFL outputs
# --------------------------------------------------

print("\n========================================")
print("PFL OUTPUT")
print("========================================")

for neuron_type in [
    "PFL1",
    "PFL2",
    "PFL3",
]:

    print(
        neuron_type,
        ":",
        network.get_type_mean_activity(
            neuron_type
        )
    )


print("\nPFL side activity:")

for neuron_type in [
    "PFL1",
    "PFL2",
    "PFL3",
]:

    print(
        neuron_type,
        "L =",
        network.get_side_mean_activity(
            neuron_type,
            "L",
        ),
        "R =",
        network.get_side_mean_activity(
            neuron_type,
            "R",
        ),
    )