"""
hemibrain_loader.py

Pulls the fly central complex (CX) navigation circuit from the hemibrain
connectome via neuPrint, and exports it to a standardized local format:

    local_cache/hemibrain_cx_neurons.csv
    local_cache/hemibrain_cx_connections.csv

This is the ONLY file in the project that talks to the neuPrint API.
Everything downstream (the simulator, environment, evaluation code)
only ever reads the two CSV files produced here, and never needs to
know this data came from hemibrain specifically. This is what lets you
swap in a flywire_loader.py later without touching any other code.

Run this once. After it succeeds, you don't need internet access
again for the rest of the project (including your demo).

Usage:
    python hemibrain_loader.py --token YOUR_NEUPRINT_TOKEN

Or set the token as an environment variable:
    export NEUPRINT_TOKEN=your_token_here
    python hemibrain_loader.py
"""

import os
import sys
import argparse
import pandas as pd

from neuprint import Client, fetch_neurons, fetch_adjacencies, NeuronCriteria as NC


# ---------------------------------------------------------------------------
# Configuration: the CX cell types that make up the ring-attractor /
# heading-and-steering circuit. These are published, named cell types —
# not manually hand-picked neurons — so this selection is defensible and
# reproducible by anyone re-running this script.
# ---------------------------------------------------------------------------

CX_CELL_TYPES = [
    # Ring neurons — visual/sensory input to the compass
    "ER1_a", "ER1_b", "ER2_a", "ER2_b", "ER2_c", "ER2_d",
    "ER3a_a", "ER3a_b", "ER3d_a", "ER3d_b", "ER3d_c",
    "ER3w_a", "ER3w_b", "ER3m", "ER3p_a", "ER3p_b",
    "ER4d", "ER4m", "ER5",
    # Compass neurons
    "EPG", "EPGt",
    # Angular velocity / rotation neurons
    "PEN_a(PEN1)", "PEN_b(PEN2)",
    # Inhibitory structural neurons for the attractor
    "Delta7",
    # Goal-direction / steering output neurons
    "PFNd", "PFNv", "PFNa_a", "PFNa_b", "PFNp_a", "PFNp_b", "PFNp_c", "PFNp_d", "PFNp_e", "PFNm_a", "PFNm_b",
    "PFL1", "PFL2", "PFL3",
]

DATASET = "hemibrain:v1.2.1"
SERVER = "neuprint.janelia.org"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "local_cache")
NEURONS_OUT = os.path.join(OUTPUT_DIR, "hemibrain_cx_neurons.csv")
CONNECTIONS_OUT = os.path.join(OUTPUT_DIR, "hemibrain_cx_connections.csv")


def get_token(cli_token):
    token = cli_token or os.environ.get("NEUPRINT_TOKEN")
    if not token:
        sys.exit(
            "ERROR: no neuPrint token provided.\n"
            "Pass --token YOUR_TOKEN, or set the NEUPRINT_TOKEN environment variable.\n"
            "Get a free token from your account page at https://neuprint.janelia.org"
        )
    return token


def fetch_cx_circuit(client):
    """Query neuPrint for the CX cell types and their connectivity."""
    print(f"Querying {len(CX_CELL_TYPES)} CX cell types from {DATASET} ...")

    criteria = NC(type=CX_CELL_TYPES, client=client)
    neuron_df, roi_df = fetch_neurons(criteria, client=client)

    if len(neuron_df) == 0:
        sys.exit(
            "ERROR: no neurons returned. Check that cell type names match this "
            "hemibrain version exactly (they can change slightly between releases)."
        )

    print(f"Found {len(neuron_df)} neurons across {neuron_df['type'].nunique()} types.")

    print("Fetching synaptic connectivity between these neurons ...")
    # NOTE: fetch_adjacencies returns (neuron_df, roi_connectivity_df) — the
    # connectivity table is the SECOND element, not the first.
    _, roi_conn_df = fetch_adjacencies(sources=neuron_df, targets=neuron_df, client=client)

    # roi_conn_df has one row per (source, target, roi) — a neuron pair
    # connected across multiple brain regions appears multiple times.
    # Collapse to one edge per (source, target) pair by summing weight
    # across regions, since the rate-model treats each pair as a single
    # coupling strength.
    conn_df = (
        roi_conn_df.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"]
        .sum()
    )

    print(f"Found {len(conn_df)} unique connected neuron pairs "
          f"(from {len(roi_conn_df)} region-level synapse records).")
    return neuron_df, conn_df


def normalize_neurons(neuron_df):
    """
    Reduce the raw neuPrint neuron table to a standardized schema that any
    future loader (e.g. flywire_loader.py) should also produce:

        neuron_id, type, instance, side, predicted_nt, predicted_nt_conf
    """
    out = pd.DataFrame({
        "neuron_id": neuron_df["bodyId"],
        "type": neuron_df["type"],
        "instance": neuron_df.get("instance", neuron_df["type"]),
        "side": neuron_df.get("somaSide", "unknown"),
        "predicted_nt": neuron_df.get("predictedNt", "unknown"),
        "predicted_nt_conf": neuron_df.get("predictedNtProb", 0.0),
    })
    return out


# ---------------------------------------------------------------------------
# Neurotransmitter sign, assigned BY CELL TYPE using documented identities
# from the literature (Hulse et al. 2021; hemibrain NT predictions,
# Eckstein et al. 2020), rather than a per-neuron API field.
#
# NOTE: an earlier version of this script tried to read a per-neuron
# 'predictedNt' field from fetch_neurons() and derive sign from that.
# That field is not reliably populated this way and returned 'unknown'
# for every neuron, silently zeroing out the entire weight matrix. This
# type-level mapping is a known, defensible, reproducible substitute --
# state this simplification explicitly in your report.
# ---------------------------------------------------------------------------

TYPE_SIGN_MAP = {
    # Compass / relay neurons -- cholinergic, excitatory
    "EPG": 1, "EPGt": 1,
    "PEN_a(PEN1)": 1, "PEN_b(PEN2)": 1,
    # Steering / output neurons -- cholinergic, excitatory
    "PFNd": 1, "PFNv": 1, "PFNa_a": 1, "PFNa_b": 1,
    "PFNp_a": 1, "PFNp_b": 1, "PFNp_c": 1, "PFNp_d": 1, "PFNp_e": 1,
    "PFNm_a": 1, "PFNm_b": 1,
    "PFL1": 1, "PFL2": 1, "PFL3": 1,
    # Structural inhibitory neuron of the attractor -- GABAergic
    "Delta7": -1,
    # Ring (visual input) neurons -- predominantly GABAergic onto EPG
    "ER1_a": -1, "ER1_b": -1, "ER2_a": -1, "ER2_b": -1, "ER2_c": -1, "ER2_d": -1,
    "ER3a_a": -1, "ER3a_b": -1, "ER3d_a": -1, "ER3d_b": -1, "ER3d_c": -1,
    "ER3w_a": -1, "ER3w_b": -1, "ER3m": -1, "ER3p_a": -1, "ER3p_b": -1,
    "ER4d": -1, "ER4m": -1, "ER5": -1,
}


def normalize_connections(conn_df, neuron_lookup):
    """
    Reduce the raw connectivity table to a standardized schema:

        source_id, target_id, weight, sign, source_type, target_type

    'weight' is the raw synapse count, min-max normalized to [0, 1] within
    this circuit so it's usable directly as a coupling strength in the
    rate model. 'sign' comes from the presynaptic neuron's TYPE, via
    TYPE_SIGN_MAP (documented cell-type neurotransmitter identity),
    not a per-neuron predicted field.
    """
    out = pd.DataFrame({
        "source_id": conn_df["bodyId_pre"],
        "target_id": conn_df["bodyId_post"],
        "raw_synapse_count": conn_df["weight"],
    })

    # Attach source neurotransmitter prediction to determine sign
    out = out.merge(
        neuron_lookup[["neuron_id", "predicted_nt", "type"]],
        left_on="source_id", right_on="neuron_id", how="left",
    ).rename(columns={"predicted_nt": "source_nt", "type": "source_type"}).drop(columns=["neuron_id"])

    out = out.merge(
        neuron_lookup[["neuron_id", "type"]],
        left_on="target_id", right_on="neuron_id", how="left",
    ).rename(columns={"type": "target_type"}).drop(columns=["neuron_id"])

    out["sign"] = out["source_type"].map(TYPE_SIGN_MAP)
    unmapped = out["sign"].isna().sum()
    if unmapped:
        unmapped_types = out.loc[out["sign"].isna(), "source_type"].unique()
        print(f"[WARNING] {unmapped} connections have a source_type not in TYPE_SIGN_MAP: "
              f"{list(unmapped_types)}. Defaulting these to excitatory (+1) -- "
              f"add them to TYPE_SIGN_MAP explicitly if you know their real identity.")
        out["sign"] = out["sign"].fillna(1)
    out["sign"] = out["sign"].astype(int)

    # Normalize synapse counts to [0, 1] as a coupling-strength proxy.
    max_count = out["raw_synapse_count"].max()
    out["weight"] = out["raw_synapse_count"] / max_count if max_count > 0 else 0.0

    out = out[["source_id", "target_id", "weight", "sign", "raw_synapse_count", "source_type", "target_type"]]
    return out


def main():
    parser = argparse.ArgumentParser(description="Pull CX circuit from hemibrain via neuPrint.")
    parser.add_argument("--token", default=None, help="neuPrint API token")
    args = parser.parse_args()

    token = get_token(args.token)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    client = Client(SERVER, dataset=DATASET, token=token)
    print(f"Connected. neuPrint version: {client.fetch_version()}")

    neuron_df, conn_df = fetch_cx_circuit(client)

    neurons_norm = normalize_neurons(neuron_df)
    connections_norm = normalize_connections(conn_df, neurons_norm)

    neurons_norm.to_csv(NEURONS_OUT, index=False)
    connections_norm.to_csv(CONNECTIONS_OUT, index=False)

    print("\nDone.")
    print(f"  Neurons     -> {NEURONS_OUT}  ({len(neurons_norm)} rows)")
    print(f"  Connections -> {CONNECTIONS_OUT}  ({len(connections_norm)} rows)")
    print("\nType breakdown:")
    print(neurons_norm["type"].value_counts())
    print("\nYou can now run the simulator entirely offline using these two files.")


if __name__ == "__main__":
    main()