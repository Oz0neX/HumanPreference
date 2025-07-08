import sys
from imitation.data.rollout import flatten_trajectories
from imitation.algorithms import bc
from train import extract_data
import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy

maps_data = extract_data()

if not maps_data:
    print("No trajectory data was found. Exiting.")
    sys.exit(1)

def train(transitions, seed):
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
        "horizon": 300
    })

    # Create a custom policy with more layers (4 layers of 128 units each)
    custom_policy = ActorCriticPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: 3e-4,
        net_arch=[128, 128, 128, 128]  # 4 hidden layers with 128 units each
    )
    
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=np.random.default_rng(0),
        policy=custom_policy,
        batch_size=100,
    )

    bc_trainer.train(n_epochs=200)
    evaluate_policy(bc_trainer.policy, env, n_eval_episodes=1)
    env.close()


for map_name, trajectories_dict in maps_data.items():
    seed = int(map_name.split('_')[1])
    
    optimal_trajectories = trajectories_dict.get('optimal', [])
    optimal_transitions = flatten_trajectories(optimal_trajectories)
    train(optimal_transitions, seed)
    
    teaching_trajectories = trajectories_dict.get('teaching', [])
    teaching_transitions = flatten_trajectories(teaching_trajectories)
    train(teaching_transitions, seed)
