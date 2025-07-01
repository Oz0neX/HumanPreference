import numpy as np
import gymnasium as gym
import sys
import os
import glob
import pandas as pd
from datetime import datetime

from imitation.algorithms import bc
from imitation.data import types

batch_size = 10
epochs = 10
state_cols = ['vehicle_x', 'vehicle_y', 'vehicle_heading', 'vehicle_speed']
action_cols = ['forward', 'left', 'right', 'brake']

def extract_data():
    maps_data = {}
    
    if not os.path.exists("trajectory_data"):
        return maps_data
    
    # Load from map_* folders
    map_paths = glob.glob(os.path.join("trajectory_data", "map_*"))
    for map_path in map_paths:
        map_name = os.path.basename(map_path)
        maps_data[map_name] = {'optimal': [], 'teaching': []}
        
        optimal_dir = os.path.join(map_path, "optimal")
        if os.path.exists(optimal_dir):
            for file_path in glob.glob(os.path.join(optimal_dir, "*.csv")):
                try:
                    df = pd.read_csv(file_path)
                    maps_data[map_name]['optimal'].append({
                        'states': df[state_cols].values,
                        'actions': df[action_cols].values,
                        'file': os.path.basename(file_path)
                    })
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        teaching_dir = os.path.join(map_path, "teaching")
        if os.path.exists(teaching_dir):
            for file_path in glob.glob(os.path.join(teaching_dir, "*.csv")):
                try:
                    df = pd.read_csv(file_path)
                    maps_data[map_name]['teaching'].append({
                        'states': df[state_cols].values,
                        'actions': df[action_cols].values,
                        'file': os.path.basename(file_path)
                    })
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    return maps_data

def process(trajectories, model_name):
    all_states = np.vstack([traj['states'] for traj in trajectories])
    all_actions = np.vstack([traj['actions'] for traj in trajectories])

    print("HARHAHRHHRHARHHRH HERE WE GO ALL STATES")
    print(all_states) # To visualize how they are being v_stacked and see if it's good for our imitation library
    demos = types.TransitionMinimal(
        obs=all_states,
        acts=all_actions,
        infos=np.array([{} for _ in range(batch_size)])
    ) #Using a transitionminimal instead of transition

    bc_trainer = bc.BC(
        observation_space=gym.spaces.Box(low=[-np.inf, -np.inf, -np.pi * 2, 0], high=[np.inf, np.inf, np.pi * 2, np.inf], shape=(4,), ),
        action_space=gym.spaces.MultiDiscrete([2, 2, 2, 2]),
        demonstrations=demos,
        rng=np.random.default_rng(0),
        batch_size=batch_size
    )
    bc_trainer.train(n_epochs=epochs)


def main():
    maps_data = extract_data()
    if not maps_data: sys.exit()

    model_name = f"_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    for map_name, trajectories in maps_data.items():
        model_name = f"optimal_{map_name}"
        process(trajectories['optimal'], "optimal" + model_name)

        model_name = f"teaching_{map_name}"
        process(trajectories['teaching'], "teach" + model_name)

main()