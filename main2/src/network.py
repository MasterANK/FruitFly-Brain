import numpy as np
import pandas as pd


class MaleCNSNetwork:

    def __init__(
        self,
        neuron_file,
        connection_file,
        dt=0.01,
        tau=0.5,
        gain=0.75,
        max_rate=1.0,
    ):

        self.dt = dt
        self.tau = tau
        self.gain = gain
        self.max_rate = max_rate

        # ==================================================
        # LOAD NEURONS
        # ==================================================

        self.neurons = pd.read_csv(neuron_file)

        self.neuron_ids = (
            self.neurons["bodyId"]
            .astype(int)
            .tolist()
        )

        self.neuron_index = {
            body_id: i
            for i, body_id in enumerate(self.neuron_ids)
        }

        self.num_neurons = len(self.neuron_ids)

        # ==================================================
        # NEUROTRANSMITTER SIGNS
        # ==================================================
        #
        # consensusNt contains the consensus neurotransmitter
        # annotation for the neurons in our dataset.
        #
        # Acetylcholine -> positive/excitatory (+1)
        # Glutamate     -> negative/inhibitory (-1)
        #
        # This is a simplified computational abstraction.
        # ==================================================

        self.neuron_sign = {}

        for _, neuron in self.neurons.iterrows():

            body_id = int(neuron["bodyId"])

            neurotransmitter = str(
                neuron["consensusNt"]
            ).strip().lower()

            if neurotransmitter == "acetylcholine":

                self.neuron_sign[body_id] = 1.0

            elif neurotransmitter == "glutamate":

                self.neuron_sign[body_id] = -1.0

            else:

                # Safe fallback for unexpected/missing values
                self.neuron_sign[body_id] = 1.0

        # ==================================================
        # LOAD CONNECTIONS
        # ==================================================

        connections = pd.read_csv(connection_file)

        self.W = np.zeros(
            (
                self.num_neurons,
                self.num_neurons,
            ),
            dtype=np.float32,
        )

        for _, connection in connections.iterrows():

            source = int(connection["bodyId_pre"])
            target = int(connection["bodyId_post"])

            weight = float(connection["weight"])

            if source not in self.neuron_index:
                continue

            if target not in self.neuron_index:
                continue

            j = self.neuron_index[source]
            i = self.neuron_index[target]

            # --------------------------------------------------
            # Apply neurotransmitter sign of the SOURCE neuron
            # --------------------------------------------------

            sign = self.neuron_sign.get(
                source,
                1.0,
            )

            self.W[i, j] += weight

        # ==================================================
        # NORMALIZE
        # ==================================================

        max_weight = np.max(
            np.abs(self.W)
        )

        if max_weight > 0:
            self.W /= max_weight

        # ==================================================
        # ACTIVITY
        # ==================================================

        self.r = np.zeros(
            self.num_neurons,
            dtype=np.float32,
        )

        self.input = np.zeros(
            self.num_neurons,
            dtype=np.float32,
        )

        # ==================================================
        # TYPE LOOKUP
        # ==================================================

        self.type_to_indices = {}

        for index, neuron_type in enumerate(
            self.neurons["type"]
        ):

            if pd.isna(neuron_type):
                continue

            self.type_to_indices.setdefault(
                neuron_type,
                [],
            ).append(index)

        # ==================================================
        # NETWORK INFORMATION
        # ==================================================

        print("========================================")
        print("MALECNS NETWORK")
        print("========================================")

        print(
            "Neurons:",
            self.num_neurons,
        )

        print(
            "Non-zero connections:",
            np.count_nonzero(self.W),
        )

        # Show neurotransmitter statistics
        print(
            "Acetylcholine neurons:",
            sum(
                1
                for sign in self.neuron_sign.values()
                if sign == 1.0
            ),
        )

        print(
            "Glutamate neurons:",
            sum(
                1
                for sign in self.neuron_sign.values()
                if sign == -1.0
            ),
        )

    # ======================================================
    # ACTIVATION
    # ======================================================

    def activation(self, x):

        return np.tanh(x)

    # ======================================================
    # RESET
    # ======================================================

    def reset(self):

        self.r[:] = 0
        self.input[:] = 0

    # ======================================================
    # INPUT BY BODY ID
    # ======================================================

    def set_neuron_input(
        self,
        body_id,
        value,
    ):

        if body_id not in self.neuron_index:
            return

        index = self.neuron_index[body_id]

        self.input[index] = value

    # ======================================================
    # INPUT BY TYPE
    # ======================================================

    def set_type_input(
        self,
        neuron_type,
        value,
    ):

        indices = self.type_to_indices.get(
            neuron_type,
            [],
        )

        if not indices:
            return

        self.input[indices] = value

    # ======================================================
    # INPUT BY TYPE + SIDE
    # ======================================================

    def set_side_input(
        self,
        neuron_type,
        side,
        value,
    ):

        if "somaSide" not in self.neurons.columns:
            return

        for index in self.type_to_indices.get(
            neuron_type,
            [],
        ):

            neuron_side = self.neurons.iloc[
                index
            ]["somaSide"]

            if neuron_side == side:

                self.input[index] = value

    # ======================================================
    # CLEAR INPUT
    # ======================================================

    def clear_input(self):

        self.input[:] = 0

    # ======================================================
    # SET INPUT
    # ======================================================
    def set_input(self, inputs):
        """
        Set external input to individual neurons.

        Example:

            network.set_input({
                0: 0.5
            })

        Here 0 refers to neuron index 0.

        Multiple neurons can be stimulated:

            network.set_input({
                0: 0.5,
                10: 0.2,
                25: 0.8
            })
        """

        self.clear_input()

        for neuron_index, value in inputs.items():

            neuron_index = int(neuron_index)

            if neuron_index < 0:
                continue

            if neuron_index >= self.num_neurons:
                continue

            self.input[neuron_index] = float(value)

    # ======================================================
    # STEP
    # ======================================================

    def step(self):

        recurrent_input = self.W @ self.r

        total_input = (
            self.gain * recurrent_input
            + self.input
        )

        target_activity = self.activation(
            total_input
        )

        dr = (
            self.dt
            / self.tau
            * (
                -self.r
                + target_activity
            )
        )

        self.r += dr

        self.r = np.clip(
            self.r,
            0,
            self.max_rate,
        )

        return self.r.copy()

    # ======================================================
    # GET ACTIVITY OF A TYPE
    # ======================================================

    def get_type_activity(
        self,
        neuron_type,
    ):

        indices = self.type_to_indices.get(
            neuron_type,
            [],
        )

        if not indices:
            return np.array([])

        return self.r[indices]

    # ======================================================
    # GET MEAN TYPE ACTIVITY
    # ======================================================

    def get_type_mean_activity(
        self,
        neuron_type,
    ):

        activity = self.get_type_activity(
            neuron_type
        )

        if len(activity) == 0:
            return 0.0

        return float(
            np.mean(activity)
        )

    # ======================================================
    # GET SIDE ACTIVITY
    # ======================================================

    def get_side_mean_activity(
        self,
        neuron_type,
        side,
    ):

        if "somaSide" not in self.neurons.columns:
            return 0.0

        values = []

        for index in self.type_to_indices.get(
            neuron_type,
            [],
        ):

            neuron_side = self.neurons.iloc[
                index
            ]["somaSide"]

            if neuron_side == side:
                values.append(
                    self.r[index]
                )

        if not values:
            return 0.0

        return float(
            np.mean(values)
        )

    # ======================================================
    # AVAILABLE TYPES
    # ======================================================

    def available_types(self):

        return sorted(
            self.type_to_indices.keys()
        )