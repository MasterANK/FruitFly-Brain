"""
used to find upstream / downastream connections
"""

from neuprint import fetch_adjacencies, NeuronCriteria

from male_cns_loader import client, NAVIGATION_TYPES


criteria = NeuronCriteria(type=NAVIGATION_TYPES)


print("========================================")
print("MALECNS NAVIGATION INTERFACE ANALYSIS")
print("========================================")


# --------------------------------------------------
# UPSTREAM
# --------------------------------------------------

print("\nFinding upstream neurons...")

upstream_info, upstream_connections = fetch_adjacencies(
    targets=criteria
)

print("Upstream neuron records:", len(upstream_info))
print("Upstream connections:", len(upstream_connections))

print("\nTop upstream cell types:")

if "type" in upstream_info.columns:
    print(
        upstream_info["type"]
        .value_counts()
        .head(30)
        .to_string()
    )


# --------------------------------------------------
# DOWNSTREAM
# --------------------------------------------------

print("\nFinding downstream neurons...")

downstream_info, downstream_connections = fetch_adjacencies(
    sources=criteria
)

print("Downstream neuron records:", len(downstream_info))
print("Downstream connections:", len(downstream_connections))

print("\nTop downstream cell types:")

if "type" in downstream_info.columns:
    print(
        downstream_info["type"]
        .value_counts()
        .head(30)
        .to_string()
    )