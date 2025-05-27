#!/usr/bin/env python
import os
import json
import argparse
import numpy as np
from metadrive.utils.config import Config
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.map.base_map import BaseMap
from metadrive.component.algorithm.BIG import BigGenerateMethod

def generate_and_save_map(output_path, map_seed=None, map_string=None, lane_width=3.5, num_lanes=3):
    """
    Generates a MetaDrive map and saves its configuration to a JSON file.
    """

    def config_to_dict_default(obj):
        """Custom JSON encoder for MetaDrive Config objects."""
        if isinstance(obj, Config):
            return obj.get_dict()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    env_config = {
        "num_scenarios": 1,
        "use_render": False,
        "log_level": 50,
    }

    current_map_config = {
        BaseMap.LANE_WIDTH: lane_width,
        BaseMap.LANE_NUM: num_lanes
    }
    if map_string:
        current_map_config[BaseMap.GENERATE_CONFIG] = map_string
        current_map_config[BaseMap.GENERATE_TYPE] = BigGenerateMethod.BLOCK_SEQUENCE
    
    env_config["map_config"] = current_map_config

    if map_seed is not None:
        env_config["start_seed"] = map_seed

    env = MetaDriveEnv(config=env_config)
    try:
        env.reset()
        map_data_to_save = env.current_map.get_meta_data()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(map_data_to_save, f, indent=4, default=config_to_dict_default)
        print(f"Map data successfully saved to: {output_path}")

    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save a MetaDrive map configuration.")
    parser.add_argument(
        "--output", 
        type=str, 
        default="c:/Users/Yasir/Desktop/Code/Experiments/trajectory_preferences/preset_maps/my_preset_map.json", 
        help="Full path to save the generated map JSON file."
    )
    parser.add_argument("--seed", type=int, default=12345, help="Seed for map generation.")
    parser.add_argument("--map_string", type=str, default="SCS", help="A string to define the map structure (e.g., 'SCSXS').")
    parser.add_argument("--lane_width", type=float, default=10.0, help="Lane width for the map.")
    parser.add_argument("--num_lanes", type=int, default=4, help="Number of lanes for the map.")

    args = parser.parse_args()

    print(f"Generating map with seed={args.seed}, map_string='{args.map_string}', lane_width={args.lane_width}, num_lanes={args.num_lanes}")
    generate_and_save_map(args.output, map_seed=args.seed, map_string=args.map_string, lane_width=args.lane_width, num_lanes=args.num_lanes)
    print("Map generation complete.")