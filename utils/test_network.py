"""
test_network.py

Validates network.py BEFORE you point it at your real hemibrain CSVs.

It builds a small synthetic connectome (same file schema your
hemibrain_loader.py produces) with a known structure:
    - a few "ER" (sensory input) neurons
    - a few "EPG" (relay) neurons, driven by ER
    - a few "PFL" (output) neurons, driven by EPG
    - one inhibitory neuron feeding back onto EPG (to check inhibition works)

Then it checks three things any correct rate-model implementation must
satisfy:
    1. With zero input, activity stays at/near zero (no spontaneous blow-up).
    2. With sustained input to ER, activity propagates downstream to PFL
       within a reasonable number of steps, and stays bounded.
    3. Removing the input causes activity to decay back down.

If all three pass, the simulator logic itself is validated, and you can
swap in your real hemibrain_cx_neurons.csv / hemibrain_cx_connections.csv
with confidence that any weird behavior you see is about the real
connectome's structure or your gain/tau tuning — not a bug in network.py.
"""

import os
import numpy as np
import pandas as pd
import sys

# 1. Get the absolute path of the shared parent folder
# (Since test_network.py is in 'utils', going up 2 levels hits the parent folder)
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

# 2. Add the parent folder to Python's search path for module imports
sys.path.append(PARENT_DIR)

# Now you can import directly from 'main'
from main.network import ConnectomeNetwork

TMP_DIR = os.path.join(os.path.dirname(__file__), "_synthetic_test_data")


def build_synthetic_connectome():
    os.makedirs(TMP_DIR, exist_ok=True)

    neurons = []
    for i in range(5):
        neurons.append({"neuron_id": 100 + i, "type": "ER", "instance": "ER",
                         "side": "R", "predicted_nt": "acetylcholine", "predicted_nt_conf": 0.9})
    for i in range(5):
        neurons.append({"neuron_id": 200 + i, "type": "EPG", "instance": "EPG",
                         "side": "R", "predicted_nt": "acetylcholine", "predicted_nt_conf": 0.9})
    for i in range(3):
        neurons.append({"neuron_id": 300 + i, "type": "PFL", "instance": "PFL",
                         "side": "R", "predicted_nt": "acetylcholine", "predicted_nt_conf": 0.9})
    # one inhibitory neuron
    neurons.append({"neuron_id": 400, "type": "Delta7", "instance": "Delta7",
                     "side": "R", "predicted_nt": "gaba", "predicted_nt_conf": 0.9})

    neurons_df = pd.DataFrame(neurons)

    connections = []
    # ER -> EPG, excitatory
    for er_id in range(100, 105):
        for epg_id in range(200, 205):
            connections.append({"source_id": er_id, "target_id": epg_id,
                                 "weight": 0.6, "sign": 1, "raw_synapse_count": 30,
                                 "source_type": "ER", "target_type": "EPG"})
    # EPG -> PFL, excitatory
    for epg_id in range(200, 205):
        for pfl_id in range(300, 303):
            connections.append({"source_id": epg_id, "target_id": pfl_id,
                                 "weight": 0.5, "sign": 1, "raw_synapse_count": 25,
                                 "source_type": "EPG", "target_type": "PFL"})
    # Delta7 -> EPG, inhibitory feedback
    for epg_id in range(200, 205):
        connections.append({"source_id": 400, "target_id": epg_id,
                             "weight": 0.4, "sign": -1, "raw_synapse_count": 20,
                             "source_type": "Delta7", "target_type": "EPG"})
    # EPG -> Delta7, excitatory (drives the inhibitory feedback loop)
    for epg_id in range(200, 205):
        connections.append({"source_id": epg_id, "target_id": 400,
                             "weight": 0.3, "sign": 1, "raw_synapse_count": 15,
                             "source_type": "EPG", "target_type": "Delta7"})

    connections_df = pd.DataFrame(connections)

    neurons_path = os.path.join(TMP_DIR, "neurons.csv")
    connections_path = os.path.join(TMP_DIR, "connections.csv")
    neurons_df.to_csv(neurons_path, index=False)
    connections_df.to_csv(connections_path, index=False)
    return neurons_path, connections_path


def run_checks():
    neurons_path, connections_path = build_synthetic_connectome()
    net = ConnectomeNetwork(neurons_path, connections_path, dt=1.0, tau=5.0, gain=2.0)

    print(f"Loaded synthetic network: {net.n} neurons, types = {net.available_types()}")

    # --- Check 1: zero input -> stays near zero ---
    net.reset()
    for _ in range(50):
        net.step()
    resting = net.r.copy()
    assert np.all(resting < 0.05), f"FAIL: spontaneous activity without input: max={resting.max():.4f}"
    print("[PASS] Check 1: no spontaneous blow-up with zero input.")

    # --- Check 2: sustained ER input propagates to PFL and stays bounded ---
    net.reset()
    pfl_trace = []
    for t in range(200):
        net.set_input_by_type("ER", 3.0)   # strong sustained drive
        net.step()
        pfl_trace.append(net.get_output_by_type("PFL").mean())

    assert max(pfl_trace) > 0.1, (
        f"FAIL: PFL never responded to ER input (max PFL activity = {max(pfl_trace):.4f}). "
        f"Check weight signs / gain."
    )
    assert np.all(net.r <= net.max_rate + 1e-6), "FAIL: activity exceeded max_rate — unstable."
    assert not np.any(np.isnan(net.r)), "FAIL: NaNs in activity — numerical instability."
    print(f"[PASS] Check 2: input propagated ER -> EPG -> PFL "
          f"(final PFL mean activity = {pfl_trace[-1]:.4f}), stayed bounded, no NaNs.")

    # --- Check 3: removing input causes decay ---
    peak_r = net.r.copy()
    for t in range(100):
        net.clear_input()
        net.step()
    assert net.r.mean() < peak_r.mean(), "FAIL: activity did not decay after input removed."
    print(f"[PASS] Check 3: activity decayed after input removed "
          f"(peak mean={peak_r.mean():.4f} -> after decay mean={net.r.mean():.4f}).")

    # --- Check 4: tonic bias seeds a self-sustaining loop, and inhibitory
    #     ER input correctly SUPPRESSES it rather than exciting it (this
    #     mirrors the real hemibrain situation: ER->EPG is inhibitory) ---
    net2 = ConnectomeNetwork(neurons_path, connections_path, dt=1.0, tau=5.0, gain=2.0)
    net2.reset()
    net2.set_tonic_by_type("EPG", 1.5)   # seed the EPG/PFL pathway independent of ER
    for _ in range(100):
        net2.step()
    epg_with_tonic_only = net2.get_output_by_type("EPG").mean()
    assert epg_with_tonic_only > 0.1, (
        f"FAIL: tonic bias did not produce sustained EPG activity "
        f"(mean={epg_with_tonic_only:.4f})."
    )
    print(f"[PASS] Check 4a: tonic bias alone sustains EPG activity "
          f"(mean EPG activity = {epg_with_tonic_only:.4f}).")

    # Now add strong inhibitory-style input on top (simulate what an
    # inhibitory ER->EPG connection would do, by adding negative external
    # input directly, since our synthetic circuit's ER is excitatory).
    net2.set_input_by_type("ER", -10.0)
    for _ in range(50):
        net2.step()
    epg_with_inhibition = net2.get_output_by_type("EPG").mean()
    assert epg_with_inhibition < epg_with_tonic_only, (
        "FAIL: adding strong negative input did not suppress EPG activity."
    )
    print(f"[PASS] Check 4b: strong negative input suppressed EPG activity "
          f"(before={epg_with_tonic_only:.4f} -> after={epg_with_inhibition:.4f}), "
          f"confirming inhibitory drive correctly reduces activity rather than "
          f"increasing it.")

    print("\nAll checks passed. network.py logic is validated on synthetic data.")
    print("Next: apply set_tonic_by_type('EPG', ...) and set_tonic_by_type('PEN...', ...) "
          "on the REAL hemibrain circuit to seed the bump, then use ER input "
          "(which is genuinely inhibitory there) to steer it.")


if __name__ == "__main__":
    run_checks()