from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

neurons = pd.read_csv(DATA / "male_cns_navigation_neurons.csv")
connections = pd.read_csv(DATA / "male_cns_navigation_connections.csv")

circuit_ids = set(neurons["bodyId"].astype(int))

# Connections entering our circuit from neurons that are NOT in our circuit
incoming = connections[
    (~connections["bodyId_pre"].isin(circuit_ids)) &
    (connections["bodyId_post"].isin(circuit_ids))
].copy()

incoming = incoming.merge(
    neurons[["bodyId", "type"]],
    left_on="bodyId_post",
    right_on="bodyId",
    how="left"
)

print("\n========================================")
print("EXTERNAL INPUTS INTO NAVIGATION CIRCUIT")
print("========================================")

summary = (
    incoming
    .groupby("type")["weight"]
    .agg(["count", "sum"])
    .sort_values("sum", ascending=False)
)

print(summary.to_string())

print("\n========================================")
print("TOTAL EXTERNAL INPUT CONNECTIONS")
print("========================================")

print("Records:", len(incoming))
print("Target neuron types:", incoming["type"].nunique())