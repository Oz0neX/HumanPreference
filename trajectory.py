#!/usr/bin/env python
import json
import csv
import os
from datetime import datetime
from metadrive.envs.top_down_env import TopDownMetaDrive
from metadrive.component.map.base_map import BaseMap
from metadrive.examples.ppo_expert.numpy_expert import expert
os.environ['SDL_VIDEO_CENTERED'] = '1'

if __name__ == "__main__":
    PRESET_MAP_FILE_PATH = "c:/Users/Yasir/Desktop/Code/Experiments/trajectory_preferences/preset_maps/my_preset_map.json"

    try:
        with open(PRESET_MAP_FILE_PATH, 'r') as f:
            loaded_map_data = json.load(f)
        print(f"Successfully loaded preset map from: {PRESET_MAP_FILE_PATH}")
    except FileNotFoundError:
        print(f"ERROR: Preset map file not found at {PRESET_MAP_FILE_PATH}.")
        print("Please ensure the map_maker.py script has been run and the path is correct.")
        exit(1)
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from map file: {PRESET_MAP_FILE_PATH}.")
        exit(1)

    env = TopDownMetaDrive(
        dict(
            map=loaded_map_data,
            traffic_density=0.1,
            num_scenarios=1,
            start_seed=loaded_map_data["map_config"]["seed"],
            manual_control=True,
            use_render=False 
        )
    )

    base_log_dir = os.path.join(os.path.dirname(__file__), "trajectory_logs")
    os.makedirs(base_log_dir, exist_ok=True)

    current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_log_dir = os.path.join(base_log_dir, current_time_str)
    os.makedirs(run_log_dir, exist_ok=True)
    print(f"Logging trajectories to: {run_log_dir}")

    csv_header_per_episode = [
        "step_in_episode", "pos_x", "pos_y"
    ]
    episode_id_counter = 0
    current_csv_file = None
    current_csv_writer = None

    try:
        o, _ = env.reset()
        
        episode_id_counter += 1
        step_in_episode_counter = 0
        if current_csv_file:
            current_csv_file.close()
        episode_csv_path = os.path.join(run_log_dir, f"episode_{episode_id_counter}.csv")
        current_csv_file = open(episode_csv_path, 'w', newline='')
        current_csv_writer = csv.writer(current_csv_file)
        current_csv_writer.writerow(csv_header_per_episode)

        for i in range(1, 1000000):
            step_in_episode_counter += 1
            action_to_apply = expert(env.agent)
            o, r, tm, tc, info = env.step(action_to_apply)

            if step_in_episode_counter % 10 == 0:
                current_agent = env.agent
                row_data = [
                    step_in_episode_counter,
                    current_agent.position[0],
                    current_agent.position[1]
                ]
                if current_csv_writer:
                    current_csv_writer.writerow(row_data)

            env.render(mode="top_down", text={"Quit": "ESC"}, film_size=(2000, 2000))
            
            if tm or tc:
                o, _ = env.reset()
                
                episode_id_counter += 1
                step_in_episode_counter = 0
                if current_csv_file: 
                    current_csv_file.close()
                episode_csv_path = os.path.join(run_log_dir, f"episode_{episode_id_counter}.csv")
                current_csv_file = open(episode_csv_path, 'w', newline='')
                current_csv_writer = csv.writer(current_csv_file)
                current_csv_writer.writerow(csv_header_per_episode)
    finally:
        if current_csv_file and not current_csv_file.closed: 
            current_csv_file.close()
        env.close()
