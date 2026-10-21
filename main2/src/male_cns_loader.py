import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from neuprint import (
    Client,
    fetch_neurons,
    fetch_adjacencies,
    NeuronCriteria,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = "male-cns:v1.0"

NAVIGATION_TYPES = [
    # Heading representation
    "EPG",
    "EPGt",

    # Angular movement / heading update
    "PEN1",
    "PEN2",

    # Heading integration
    "Delta7",

    # PFN navigation populations
    "PFNa",
    "PFNd",
    "PFNv",
    "PFNp_a",
    "PFNp_b",
    "PFNp_c",
    "PFNp_d",
    "PFNp_e",

    # Steering / action selection
    "PFL1",
    "PFL2",
    "PFL3",

    # Related navigation population
    "PEG",
]


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

DATA_DIR.mkdir(exist_ok=True)


# ============================================================
# CONNECT TO NEUPRINT
# ============================================================

load_dotenv()

token = os.getenv("NEUPRINT_TOKEN")

if not token:
    raise RuntimeError(
        "NEUPRINT_TOKEN was not found.\n"
        "Create a .env file in the project root and add:\n"
        "NEUPRINT_TOKEN=your_token"
    )


print("========================================")
print("MALECNS NAVIGATION CIRCUIT EXTRACTION")
print("========================================")

print("\nConnecting to MaleCNS neuPrint...")

client = Client(
    "https://neuprint.janelia.org",
    dataset=DATASET,
    token=token,
)

print("Connected!")
print("Dataset:", DATASET)


# ============================================================
# FIND NAVIGATION NEURONS
# ============================================================

print("\nSearching for navigation neurons...")

criteria = NeuronCriteria(type=NAVIGATION_TYPES)

neurons, _ = fetch_neurons(criteria)


print("\n========================================")
print("NAVIGATION NEURONS")
print("========================================")

print("Total neurons:", len(neurons))

print("\nNeurons by type:")

type_counts = neurons["type"].value_counts()

for neuron_type, count in type_counts.items():
    print(f"{neuron_type:15s} : {count}")


# ============================================================
# SAVE NEURON DATA
# ============================================================

neuron_file = DATA_DIR / "male_cns_navigation_neurons.csv"

neurons.to_csv(
    neuron_file,
    index=False,
)

print("\nNeuron data saved to:")
print(neuron_file)


# ============================================================
# FETCH INTERNAL CONNECTIONS
# ============================================================

print("\nFinding connections inside navigation circuit...")

neuron_info, connections = fetch_adjacencies(
    sources=criteria,
    targets=criteria,
)


print("\n========================================")
print("NAVIGATION CONNECTIONS")
print("========================================")

print("Total connection records:", len(connections))

print("\nConnection columns:")
print(list(connections.columns))


# ============================================================
# SAVE CONNECTION DATA
# ============================================================

connection_file = DATA_DIR / "male_cns_navigation_connections.csv"

connections.to_csv(
    connection_file,
    index=False,
)

print("\nConnection data saved to:")
print(connection_file)


# ============================================================
# CREATE CELL-TYPE CONNECTIVITY SUMMARY
# ============================================================

print("\nCreating cell-type connectivity summary...")


# Map body ID → neuron type
body_to_type = dict(
    zip(
        neurons["bodyId"],
        neurons["type"],
    )
)


connections["source_type"] = connections["bodyId_pre"].map(
    body_to_type
)

connections["target_type"] = connections["bodyId_post"].map(
    body_to_type
)


# Remove anything that could not be mapped
connections = connections.dropna(
    subset=[
        "source_type",
        "target_type",
    ]
)


# Group connections by neuron type
type_connections = (
    connections
    .groupby(
        [
            "source_type",
            "target_type",
        ],
        as_index=False,
    )
    .agg(
        connection_count=("weight", "size"),
        total_weight=("weight", "sum"),
    )
    .sort_values(
        "total_weight",
        ascending=False,
    )
)


summary_file = DATA_DIR / "male_cns_navigation_type_connections.csv"

type_connections.to_csv(
    summary_file,
    index=False,
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n========================================")
print("CELL-TYPE CONNECTIVITY")
print("========================================")

print(
    type_connections.head(30).to_string(
        index=False
    )
)

print("\nCell-type summary saved to:")
print(summary_file)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========================================")
print("EXTRACTION COMPLETE")
print("========================================")

print("Navigation neurons :", len(neurons))
print("Connection records  :", len(connections))
print(
    "Cell-type pathways  :",
    len(type_connections),
)

print("\nFiles created:")

print("1.", neuron_file)
print("2.", connection_file)
print("3.", summary_file)

print("\nThe biological dataset is now stored locally.")
print("The simulator will not need neuPrint.")