from pathlib import Path

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


# --------------------------------------------------
# Load data
# --------------------------------------------------

neurons = pd.read_csv(
    DATA_DIR / "male_cns_navigation_neurons.csv"
)

connections = pd.read_csv(
    DATA_DIR / "male_cns_navigation_connections.csv"
)


print("Neuron columns:")
print(list(neurons.columns))

print("\nConnection columns:")
print(list(connections.columns))


# --------------------------------------------------
# Find ID columns automatically
# --------------------------------------------------

neuron_id_col = None

for column in ["bodyId", "bodyId_x", "body_id", "id"]:
    if column in neurons.columns:
        neuron_id_col = column
        break

if neuron_id_col is None:
    raise RuntimeError(
        "Could not find neuron ID column."
    )


source_col = None
target_col = None

possible_source = [
    "bodyId_pre",
    "bodyId_source",
    "source",
    "bodyId_x"
]

possible_target = [
    "bodyId_post",
    "bodyId_target",
    "target",
    "bodyId_y"
]

for column in possible_source:
    if column in connections.columns:
        source_col = column
        break

for column in possible_target:
    if column in connections.columns:
        target_col = column
        break


if source_col is None or target_col is None:
    raise RuntimeError(
        "\nCould not identify source/target columns.\n"
        "Look at the 'Connection columns' printed above."
    )


print("\nUsing:")
print("Neuron ID :", neuron_id_col)
print("Source    :", source_col)
print("Target    :", target_col)


# --------------------------------------------------
# Create graph
# --------------------------------------------------

G = nx.DiGraph()


# Add neurons
for _, neuron in neurons.iterrows():

    neuron_id = neuron[neuron_id_col]

    G.add_node(
        neuron_id,
        neuron_type=neuron.get("type", "unknown")
    )


# Add connections
for _, connection in connections.iterrows():

    source = connection[source_col]
    target = connection[target_col]

    G.add_edge(source, target)


print("\n========================================")
print("GRAPH")
print("========================================")

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())


# --------------------------------------------------
# Visualize
# --------------------------------------------------

plt.figure(figsize=(14, 10))

pos = nx.spring_layout(
    G,
    seed=42
)

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=80
)

nx.draw_networkx_edges(
    G,
    pos,
    arrows=True,
    arrowsize=8,
    alpha=0.4
)

plt.title("MaleCNS Navigation Circuit")

plt.axis("off")

plt.tight_layout()

plt.show()