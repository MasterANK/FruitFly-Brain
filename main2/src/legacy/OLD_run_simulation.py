from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from agent import Agent
from environment import Environment
from network import MaleCNSNetwork

# --------------------------------------------------
# PATHS
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# --------------------------------------------------
# CREATE NETWORK
# --------------------------------------------------
network = MaleCNSNetwork(
    neuron_file=DATA / "male_cns_navigation_neurons.csv",
    connection_file=DATA / "male_cns_navigation_connections.csv",
    gain=0.75,
    dt=0.01,
    tau=0.5,
)

# --------------------------------------------------
# CREATE ENVIRONMENT AND AGENT
# --------------------------------------------------
environment = Environment(
    width=100,
    height=100,
    target=(80, 60),
    seed=None,  # Set an integer (e.g. 42) for repeatable food positions.
)

agent = Agent(
    network=network,
    start=(20, 50),
    heading=0.0,
)

# --------------------------------------------------
# SIMULATION SETTINGS
# --------------------------------------------------
STEPS = 1500
EPG_INPUT = 0.2
DECISION_INTERVAL = 10
DRAW_EVERY = 3  # Update the live plot every N simulation steps.
current_turn = 0
foods_collected = 0
positions = [agent.position.copy()]
directions = []

# --------------------------------------------------
# LIVE PLOT SETUP
# --------------------------------------------------
plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))
path_line, = ax.plot([], [], linewidth=1.5, label="Fly path")
start_marker = ax.scatter(
    positions[0][0], positions[0][1], marker="o", s=70, label="Start"
)
food_marker = ax.scatter(
    environment.target[0], environment.target[1],
    marker="*", s=180, label="Food"
)
fly_marker, = ax.plot(
    [agent.position[0]], [agent.position[1]],
    marker="o", markersize=7, linestyle="None", label="Fly"
)

ax.set_xlim(0, environment.width)
ax.set_ylim(0, environment.height)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Connectome-Based Artificial Agent — live simulation")
ax.grid(True)
ax.legend(loc="upper right")
status_text = ax.text(
    0.02, 0.98, "", transform=ax.transAxes, va="top",
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
)
fig.tight_layout()
plt.show(block=False)
plt.pause(0.1)

# --------------------------------------------------
# SIMULATION LOOP
# --------------------------------------------------
for step in range(STEPS):
    # Make a new neural steering decision every DECISION_INTERVAL steps.
    if step % DECISION_INTERVAL == 0:
        direction = environment.get_target_direction(
            agent.position, agent.heading
        )
        directions.append(direction)

        network.reset()
        if direction == "LEFT":
            network.set_side_input("EPG", "L", EPG_INPUT)
        elif direction == "RIGHT":
            network.set_side_input("EPG", "R", EPG_INPUT)
        else:
            network.set_type_input("EPG", EPG_INPUT)

        for _ in range(100):
            network.step()

        current_turn = agent.get_direction()

    # Existing boundary-avoidance behavior in Agent is retained.
    agent.avoid_boundary(environment)

    if step % DECISION_INTERVAL == 0:
        agent.update(current_turn)
    else:
        agent.update()

    agent.position = environment.keep_inside(agent.position)
    positions.append(agent.position.copy())

    # Collect food and immediately spawn it at a new random location.
    if environment.reached_target(agent.position):
        foods_collected += 1
        new_target = environment.randomize_target(
            agent.position, min_distance=25, margin=5
        )
        print(
            f"Food #{foods_collected} collected at step {step + 1}; "
            f"new food at ({new_target[0]:.1f}, {new_target[1]:.1f})"
        )

    # Draw the fly and its trail while the simulation is running.
    if step % DRAW_EVERY == 0 or step == STEPS - 1:
        trail = np.asarray(positions)
        path_line.set_data(trail[:, 0], trail[:, 1])
        fly_marker.set_data([agent.position[0]], [agent.position[1]])
        food_marker.set_offsets(environment.target.reshape(1, 2))
        status_text.set_text(
            f"Step: {step + 1}/{STEPS}\n"
            f"Food collected: {foods_collected}\n"
            f"Target direction: {direction}"
        )
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.001)

print("\\n" + "=" * 40)
print("SIMULATION RESULTS")
print("=" * 40)
print(f"Steps: {len(positions) - 1}")
print(f"Food collected: {foods_collected}")
print(f"Final position: {positions[-1]}")
print(f"Current food position: {environment.target}")
print(f"Distance to current food: {final_distance:.2f}")

# Keep the completed path visible after the simulation finishes.
plt.ioff()
plt.show()
