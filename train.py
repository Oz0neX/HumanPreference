import sys
from imitation.data.rollout import flatten_trajectories
from imitation.algorithms import bc
from load_trajectories import extract_data
import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import csv

maps_data = extract_data()

if not maps_data:
    print("No trajectory data was found. Exiting.")
    sys.exit(1)

def train(transitions, seed, demo_type):
    env = MetaDriveEnv(config={
        "on_continuous_line_done": False,
        "out_of_route_done": False,
        "crash_vehicle_done": False,
        "crash_object_done": False,
        "use_render": False,
        "manual_control": False,
        "traffic_density": 0,
        "map": "SCS",
        "start_seed": seed,
        "horizon": 1000
    })

    custom_policy = ActorCriticPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        net_arch=[32, 64, 32, 6],
        lr_schedule=lambda _: 3e-4
    )
    
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=np.random.default_rng(0),
        policy=custom_policy,
        batch_size=300,
    )

    bc_trainer.train(n_epochs=300)
    by_step, by_traj, by_step_random, by_traj_random = bc_trainer.calculate_teaching_volume()
    
    all_traj_ids = sorted(by_step.keys())
    
    with open(f'teaching_volume_{demo_type}_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['trajectory_id', 'bc_timestep', 'bc_trajectory', 'random_timestep', 'random_trajectory'])
        for t_id in all_traj_ids:
            writer.writerow([
                t_id,
                round(by_step.get(t_id, 0), 5),
                round(by_traj.get(t_id, 0), 5),
                round(by_step_random.get(t_id, 0), 5),
                round(by_traj_random.get(t_id, 0), 5)
            ])
        
        writer.writerow([
            'SUM',
            round(sum(by_step.values()), 5),
            round(sum(by_traj.values()), 5),
            round(sum(by_step_random.values()), 5),
            round(sum(by_traj_random.values()), 5)
        ])
    
for map_name, trajectories_dict in maps_data.items():
    seed = int(map_name.split('_')[1])
    
    optimal_trajectories = trajectories_dict.get('optimal', [])
    optimal_transitions = flatten_trajectories(optimal_trajectories)
    train(optimal_transitions, seed, "optimal")
    
    teaching_trajectories = trajectories_dict.get('teaching', [])
    teaching_transitions = flatten_trajectories(teaching_trajectories)
    train(teaching_transitions, seed, "teaching")