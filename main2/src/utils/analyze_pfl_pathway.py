from pathlib import Path
import pandas as pd

# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

NEURONS_FILE = DATA / "male_cns_navigation_neurons.csv"
CONNECTIONS_FILE = DATA / "male_cns_navigation_connections.csv"

# ============================================================
# Load data
# ============================================================

neurons = pd.read_csv(NEURONS_FILE)
connections = pd.read_csv(CONNECTIONS_FILE)

print("\n==============================================")
print(" MaleCNS EPG → PFL PATHWAY ANALYSIS")
print("==============================================\n")
print("Neurons:", len(neurons))
print("Connections:", len(connections))

# ============================================================
# Build bodyId → type/side lookups
# ============================================================

body_to_type = dict(zip(neurons["bodyId"], neurons["type"]))
body_to_side = dict(zip(neurons["bodyId"], neurons["somaSide"]))

# ============================================================
# Add source/target type and side
# ============================================================

connections["source_type"] = connections["bodyId_pre"].map(body_to_type)
connections["target_type"] = connections["bodyId_post"].map(body_to_type)
connections["source_side"] = connections["bodyId_pre"].map(body_to_side)
connections["target_side"] = connections["bodyId_post"].map(body_to_side)

# ============================================================
# 1. Direct EPG → PFL connections
# ============================================================

print("\n==============================================")
print(" 1. DIRECT EPG → PFL CONNECTIONS")
print("==============================================\n")

direct = connections[
    (connections["source_type"] == "EPG")
    & (connections["target_type"].isin(["PFL1", "PFL2", "PFL3"]))
].copy()

if direct.empty:
    print("No direct EPG → PFL connections found.")
else:
    summary = (
        direct.groupby(["source_type", "target_type"])
        .agg(
            connection_records=("weight", "size"),
            total_weight=("weight", "sum"),
            average_weight=("weight", "mean"),
        )
        .reset_index()
        .sort_values("total_weight", ascending=False)
    )
    print(summary.to_string(index=False))

# ============================================================
# 2. EPG → intermediate neuron connections
# ============================================================

print("\n==============================================")
print(" 2. EPG DOWNSTREAM TYPES")
print("==============================================\n")

epg_outputs = connections[connections["source_type"] == "EPG"].copy()

epg_summary = (
    epg_outputs.groupby("target_type")
    .agg(
        connection_records=("weight", "size"),
        total_weight=("weight", "sum"),
    )
    .reset_index()
    .sort_values("total_weight", ascending=False)
)

print(epg_summary.head(30).to_string(index=False))

# ============================================================
# 3. Which neurons connect INTO PFL?
# ============================================================

print("\n==============================================")
print(" 3. TYPES CONNECTING INTO PFL")
print("==============================================\n")

pfl_inputs = connections[
    connections["target_type"].isin(["PFL1", "PFL2", "PFL3"])
].copy()

pfl_summary = (
    pfl_inputs.groupby(["source_type", "target_type"])
    .agg(
        connection_records=("weight", "size"),
        total_weight=("weight", "sum"),
    )
    .reset_index()
    .sort_values("total_weight", ascending=False)
)

print(pfl_summary.head(40).to_string(index=False))

# ============================================================
# 4. Direct EPG → PFL neuron pairs
# ============================================================

print("\n==============================================")
print(" 4. DIRECT EPG → PFL NEURON PAIRS")
print("==============================================\n")

if not direct.empty:
    pair_summary = (
        direct.groupby(["bodyId_pre", "bodyId_post"])
        .agg(
            total_weight=("weight", "sum"),
            connection_records=("weight", "size"),
        )
        .reset_index()
        .sort_values("total_weight", ascending=False)
    )

    pair_summary["source_side"] = pair_summary["bodyId_pre"].map(body_to_side)
    pair_summary["target_side"] = pair_summary["bodyId_post"].map(body_to_side)

    print(pair_summary.head(50).to_string(index=False))

# ============================================================
# 5. EPG → PFL connectivity by side
# ============================================================

print("\n==============================================")
print(" 5. EPG → PFL SIDE CONNECTIVITY")
print("==============================================\n")

if not direct.empty:
    side_summary = (
        direct.groupby(["source_side", "target_side", "target_type"])
        .agg(
            connection_records=("weight", "size"),
            total_weight=("weight", "sum"),
        )
        .reset_index()
        .sort_values("total_weight", ascending=False)
    )

    print(side_summary.to_string(index=False))

# ============================================================
# 6. Find simple two-step EPG → X → PFL paths
# ============================================================

print("\n==============================================")
print(" 6. TWO-STEP EPG → X → PFL PATHS")
print("==============================================\n")

# EPG outputs
epg_edges = connections[connections["source_type"] == "EPG"][
    ["bodyId_pre", "bodyId_post", "weight"]
].rename(
    columns={
        "bodyId_pre": "epg_id",
        "bodyId_post": "middle_id",
        "weight": "epg_weight",
    }
)

# Connections INTO PFL
pfl_edges = connections[
    connections["target_type"].isin(["PFL1", "PFL2", "PFL3"])
][["bodyId_pre", "bodyId_post", "weight"]].rename(
    columns={
        "bodyId_pre": "middle_id",
        "bodyId_post": "pfl_id",
        "weight": "pfl_weight",
    }
)

# Join the two edges
paths = epg_edges.merge(pfl_edges, on="middle_id", how="inner")

if paths.empty:
    print("No two-step EPG → X → PFL paths found.")
else:
    paths["middle_type"] = paths["middle_id"].map(body_to_type)
    paths["path_strength"] = paths["epg_weight"] * paths["pfl_weight"]

    path_summary = (
        paths.groupby("middle_type")
        .agg(
            path_count=("path_strength", "size"),
            total_path_strength=("path_strength", "sum"),
        )
        .reset_index()
        .sort_values("total_path_strength", ascending=False)
    )

    print(path_summary.head(30).to_string(index=False))

# ============================================================
# 7. Final interpretation
# ============================================================

print("\n==============================================")
print(" ANALYSIS COMPLETE")
print("==============================================\n")
print(
    "Use the results above to determine whether "
    "EPG has a direct or indirect route to PFL.\n"
)