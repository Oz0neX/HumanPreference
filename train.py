import sys
from imitation.data.rollout import flatten_trajectories
from imitation.algorithms import bc
from load_trajectories import extract_data
import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from scipy.stats import entropy
import torch
from imitation.data.types import Trajectory

BATCH_SIZE = 300
TRAINING_EPOCHS = 300

DISCRETE_STEERING_DIM = 5
DISCRETE_THROTTLE_DIM = 5

def discrete_values_to_indices(discrete_values, steering_dim, throttle_dim):
    steering, throttle = discrete_values[0], discrete_values[1]
    
    steering_idx = int(np.clip((steering + 1.0) / 2.0 * (steering_dim - 1), 0, steering_dim - 1))
    throttle_idx = int(np.clip((throttle + 1.0) / 2.0 * (throttle_dim - 1), 0, throttle_dim - 1))
    
    return np.array([steering_idx, throttle_idx], dtype=np.int64)

def convert_trajectory_actions_to_indices(trajectory, steering_dim=DISCRETE_STEERING_DIM, throttle_dim=DISCRETE_THROTTLE_DIM):
    if len(trajectory.acts) == 0:
        return trajectory
    
    discrete_actions = []
    for action in trajectory.acts:
        discrete_action = discrete_values_to_indices(action, steering_dim, throttle_dim)
        discrete_actions.append(discrete_action)
    
    return Trajectory(
        obs=trajectory.obs,
        acts=np.array(discrete_actions, dtype=np.int64),
        infos=trajectory.infos,
        terminal=trajectory.terminal
    )

maps_data = extract_data()
if not maps_data:
    sys.exit(1)

map_name = list(maps_data.keys())[0]
print(f"For map: {map_name}")
trajectories_dict = maps_data[map_name]
seed = int(map_name.split('_')[1])

device = "cuda" if torch.cuda.is_available() else "cpu"

# Trajectories are saved with discrete values in range [-1, 1]
optimal_trajectories = trajectories_dict.get('optimal', [])
teaching_trajectories = trajectories_dict.get('teaching', [])

# Convert discrete values to indices for MultiDiscrete action space
optimal_trajectories_indices = [convert_trajectory_actions_to_indices(traj) for traj in optimal_trajectories]
teaching_trajectories_indices = [convert_trajectory_actions_to_indices(traj) for traj in teaching_trajectories]

optimal_transitions = flatten_trajectories(optimal_trajectories_indices)
teaching_transitions = flatten_trajectories(teaching_trajectories_indices)

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
    "horizon": 1000,
    "discrete_action": True,
    "discrete_steering_dim": DISCRETE_STEERING_DIM,
    "discrete_throttle_dim": DISCRETE_THROTTLE_DIM,
    "use_multi_discrete": True
})

def create_policy():
    return ActorCriticPolicy(
        observation_space=env.observation_space, action_space=env.action_space,
        net_arch=[32, 64, 32, 6], lr_schedule=lambda _: 3e-4
    )

def calculate_tv_likelihood(bc_tv_dict, random_tv_dict):
    bc_tvs = np.array(list(bc_tv_dict.values()))
    bc_tvs = -bc_tvs
    rand_tvs = np.array(list(random_tv_dict.values()))
    rand_tvs = -rand_tvs

    sum_exp_random_tv = np.sum(np.exp(rand_tvs))
    
    likelihoods = []
    for bc_tv in bc_tvs:
        exp_bc_tv = np.exp(bc_tv)
        p_i = exp_bc_tv / (exp_bc_tv + sum_exp_random_tv)
        likelihoods.append(p_i)
    
    result = 0
    for v in likelihoods:
        result += v
        if result == 0: result = v

    result /= len(likelihoods)
    return result

optimal_model = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=optimal_transitions,
    rng=np.random.default_rng(0),
    policy=create_policy(),
    batch_size=BATCH_SIZE
)
eval_grad_norm_opt = optimal_model.train(n_epochs=TRAINING_EPOCHS)

teaching_model = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=teaching_transitions,
    rng=np.random.default_rng(0),
    policy=create_policy(),
    batch_size=BATCH_SIZE
)
eval_grad_norm_teach = teaching_model.train(n_epochs=TRAINING_EPOCHS)

optimal_demos = optimal_model._demo_data_loader
teaching_demos = teaching_model._demo_data_loader

tv_opt_on_opt = optimal_model.calculate_teaching_volume(eval_grad_norm_opt, optimal_demos)
tv_opt_on_teach = optimal_model.calculate_teaching_volume(eval_grad_norm_opt, teaching_demos)

tv_teach_on_teach = teaching_model.calculate_teaching_volume(eval_grad_norm_teach, teaching_demos)
tv_teach_on_opt = teaching_model.calculate_teaching_volume(eval_grad_norm_teach, optimal_demos)

results = {}
results['tv_likelihood_timestep'] = {
    'opt_on_opt': calculate_tv_likelihood(tv_opt_on_opt[0], tv_opt_on_opt[2]),
    'opt_on_teach': calculate_tv_likelihood(tv_opt_on_teach[0], tv_opt_on_teach[2]),
    'teach_on_teach': calculate_tv_likelihood(tv_teach_on_teach[0], tv_teach_on_teach[2]),
    'teach_on_opt': calculate_tv_likelihood(tv_teach_on_opt[0], tv_teach_on_opt[2]),
}
results['tv_likelihood_trajectory'] = {
    'opt_on_opt': calculate_tv_likelihood(tv_opt_on_opt[1], tv_opt_on_opt[3]),
    'opt_on_teach': calculate_tv_likelihood(tv_opt_on_teach[1], tv_opt_on_teach[3]),
    'teach_on_teach': calculate_tv_likelihood(tv_teach_on_teach[1], tv_teach_on_teach[3]),
    'teach_on_opt': calculate_tv_likelihood(tv_teach_on_opt[1], tv_teach_on_opt[3]),
}
results['direct_log_likelihood'] = {
    'opt_on_opt': tv_opt_on_opt[4],
    'opt_on_teach': tv_opt_on_teach[4],
    'teach_on_teach': tv_teach_on_teach[4],
    'teach_on_opt': tv_teach_on_opt[4],
}

print("\nTeaching Score probabilities")
print("-"*30)
for key, val in results['tv_likelihood_timestep'].items():
    print(f"{key:<15}: {val:.8f}")
    
print("\nLog-Likelihood")
print("-"*30)
for key, val in results['direct_log_likelihood'].items():
    print(f"{key:<15}: {val:.3f}")

print("\n" + "="*60)