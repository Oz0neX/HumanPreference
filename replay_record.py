#!/usr/bin/env python3
"""
Simple MetaDrive Trajectory Recording and Replay
Record a human-driven trajectory and immediately replay it.
"""

import numpy as np
import os
import time
import sys
from datetime import datetime
from metadrive.envs.metadrive_env import MetaDriveEnv
from imitation.data import types
from imitation.data import serialize


def continuous_to_discrete_indices(continuous_action, steering_dim=5, throttle_dim=5):
    """Convert continuous action to discrete indices (0-4 for each dimension)"""
    steering, throttle = continuous_action[0], continuous_action[1]

    # Convert steering from [-1, 1] to [0, 4]
    steering_idx = round((steering + 1.0) / 2.0 * (steering_dim - 1))
    steering_idx = np.clip(steering_idx, 0, steering_dim - 1)

    # Convert throttle from [-1, 1] to [0, 4]
    throttle_idx = round((throttle + 1.0) / 2.0 * (throttle_dim - 1))
    throttle_idx = np.clip(throttle_idx, 0, throttle_dim - 1)

    # Return as tuple for multi-discrete action space
    return (int(steering_idx), int(throttle_idx))


class ReplayPolicy:
    def __init__(self, trajectory):
        self.trajectory = trajectory
        self.step = 0

    def act(self):
        if self.step >= len(self.trajectory.acts):
            return [0.0, 0.0]
        action = self.trajectory.acts[self.step]
        self.step += 1
        return action


def main(iteration_num=1, seed=12345):
    config = {
        "map": "SCS",
        "traffic_density": 0.1,
        "start_seed": seed,
        "manual_control": True,
        "use_render": True,
        "vehicle_config": {"show_navi_mark": False},
        "discrete_action": True,
        "discrete_steering_dim": 5,
        "discrete_throttle_dim": 5,
        "use_multi_discrete": True,
        "on_continuous_line_done": False,
        "out_of_route_done": False,
        "crash_vehicle_done": False,
        "crash_object_done": False,
    }

    print("Starting recording phase...")

    # RECORDING PHASE
    env = MetaDriveEnv(config)
    obs, info = env.reset()
    observations = []
    actions = []

    print("Drive the vehicle. Close window when done recording.")

    while True:
        obs, reward, terminated, truncated, info = env.step([0, 0])

        steering = info.get('steering', 0.0)
        throttle = info.get('acceleration', 0.0)
        continuous_action = np.array([steering, throttle], dtype=np.float32)

        observations.append(obs)

        # Only add action if we have a previous observation (trajectory format requires: obs0, act0, obs1, act1, ..., obsN)
        if len(observations) > 1:
            actions.append(prev_continuous_action)

        prev_continuous_action = continuous_action

        if terminated or truncated:
            break

    env.close()
    print(f"Recorded {len(observations)} observations and {len(actions)} actions")

    # Create trajectory (should have one more observation than actions)
    trajectory = types.Trajectory(
        obs=np.array(observations),
        acts=np.array(actions),
        infos=np.array([{}] * len(actions)),
        terminal=True
    )

    # Save trajectory to /recorded folder with timestamp naming
    recorded_dir = "recorded"
    os.makedirs(recorded_dir, exist_ok=True)

    # Use timestamp naming like original experiment code
    timestamp = datetime.now().strftime("%m%d_%H%M%S_%f")[:-3]
    save_path = os.path.join(recorded_dir, f"trajectory_{timestamp}")

    serialize.save(save_path, [trajectory])
    print(f"Trajectory saved to {save_path}")

    # COMMENTED OUT: Replay functionality
    # Wait 10 seconds before replay
    print("Waiting 10 seconds before replay...")
    time.sleep(10)

    # Load the saved trajectory
    print("Loading saved trajectory...")
    loaded_trajectories = serialize.load(save_path)
    trajectory = loaded_trajectories[0]
    print(f"Loaded trajectory with {len(trajectory.obs)} observations and {len(trajectory.acts)} actions")

    print("Starting replay phase...")

    # REPLAY PHASE - Use continuous actions instead of discrete
    config["manual_control"] = False
    config["discrete_action"] = False
    config["use_multi_discrete"] = False
    env = MetaDriveEnv(config)
    obs, info = env.reset()

    policy = ReplayPolicy(trajectory)

    print("Replaying trajectory... Close window when done.")

    while True:
        action = policy.act()
        obs, reward, terminated, truncated, info = env.step(action)

        if policy.step >= len(trajectory.acts) or terminated or truncated:
            break

    env.close()
    print("Replay complete!")

    print("Recording complete! Trajectory saved.")


if __name__ == "__main__":
    iteration_num = 1
    seed = 12345

    if len(sys.argv) >= 2:
        try:
            iteration_num = int(sys.argv[1])
        except ValueError:
            print(f"Invalid iteration number: {sys.argv[1]}, using default 1")

    if len(sys.argv) >= 3:
        try:
            seed = int(sys.argv[2])
        except ValueError:
            print(f"Invalid seed: {sys.argv[2]}, using default 12345")

    main(iteration_num, seed)
