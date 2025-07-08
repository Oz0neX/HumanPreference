import sys
from imitation.data.rollout import flatten_trajectories
from imitation.algorithms import bc
from train import extract_data
import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv
from stable_baselines3.common.evaluation import evaluate_policy

maps_data = extract_data()

if not maps_data:
    print("No trajectory data was found. Exiting.")
    sys.exit(1)

def train(transitions, seed):
    env = MetaDriveEnv(config={
        "crash_vehicle_done": False,
        "crash_object_done": False,
        "crash_human_done": False,
        "out_of_road_done": False,
        "use_render": True,
        "manual_control": False,
        "traffic_density": 0.1,
        "map": "SCS",
        "start_seed": seed,
        "horizon": 1000
    })

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=np.random.default_rng(0),
    )

    bc_trainer.train(n_epochs=1)

for map_name, trajectories_dict in maps_data.items():
    seed = int(map_name.split('_')[1])
    
    optimal_trajectories = trajectories_dict.get('optimal', [])
    optimal_transitions = flatten_trajectories(optimal_trajectories)
    train(optimal_transitions, seed)
    
    teaching_trajectories = trajectories_dict.get('teaching', [])
    teaching_transitions = flatten_trajectories(teaching_trajectories)
    train(teaching_transitions, seed)
