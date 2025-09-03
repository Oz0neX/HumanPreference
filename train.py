import sys
from imitation.data.rollout import flatten_trajectories
from imitation.algorithms import bc
from load_trajectories import extract_data
import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
from scipy.stats import entropy

BATCH_SIZE = 300
TRAINING_EPOCHS = 300

maps_data = extract_data()
if not maps_data:
    sys.exit(1)

map_name = list(maps_data.keys())[0]
trajectories_dict = maps_data[map_name]
seed = int(map_name.split('_')[1])

optimal_transitions = flatten_trajectories(trajectories_dict.get('optimal', []))
teaching_transitions = flatten_trajectories(trajectories_dict.get('teaching', []))

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
    "discrete_action": True
})

def create_policy():
    return ActorCriticPolicy(
        observation_space=env.observation_space, action_space=env.action_space,
        net_arch=[32, 64, 32, 6], lr_schedule=lambda _: 3e-4
    )

def calculate_distribution(bc_tv_dict, random_tv_dict):
    bc_tvs = np.array(list(bc_tv_dict.values()))
    rand_tvs = np.array(list(random_tv_dict.values()))
    sum_exp_bc = np.sum(np.exp(bc_tvs))
    sum_exp_rand = np.sum(np.exp(rand_tvs))
    denominator = sum_exp_bc + sum_exp_rand
    raw_likelihoods = np.exp(bc_tvs) / denominator
    final_dist = raw_likelihoods / np.sum(raw_likelihoods)
    return final_dist

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
tv_opt_on_teach = optimal_model.calculate_teaching_volume(eval_grad_norm_opt, optimal_demos)

tv_teach_on_teach = teaching_model.calculate_teaching_volume(eval_grad_norm_teach, teaching_demos)
tv_teach_on_opt = teaching_model.calculate_teaching_volume(eval_grad_norm_teach, teaching_demos)

results = {}

# Data format tv_type_on_type[0, 1, 2, 3]
# tv_type_on_type[0]: BC Timestep TV
# tv_type_on_type[1]: BC Trajectory TV
# tv_type_on_type[2]: Random Timestep TV
# tv_type_on_type[3]: Random Trajectory TV

# By timestep
dist_opt_on_opt_ts = calculate_distribution(tv_opt_on_opt[0], tv_opt_on_opt[2])
dist_opt_on_teach_ts = calculate_distribution(tv_opt_on_teach[0], tv_opt_on_teach[2])
results['kl_opt_model_timestep'] = entropy(dist_opt_on_opt_ts, dist_opt_on_teach_ts)

dist_teach_on_teach_ts = calculate_distribution(tv_teach_on_teach[0], tv_teach_on_teach[2])
dist_teach_on_opt_ts = calculate_distribution(tv_teach_on_opt[0], tv_teach_on_opt[2])
results['kl_teach_model_timestep'] = entropy(dist_teach_on_teach_ts, dist_teach_on_opt_ts)

# By trajectory
dist_opt_on_opt_tj = calculate_distribution(tv_opt_on_opt[1], tv_opt_on_opt[3])
dist_opt_on_teach_tj = calculate_distribution(tv_opt_on_teach[1], tv_opt_on_teach[3])
results['kl_opt_model_trajectory'] = entropy(dist_opt_on_opt_tj, dist_opt_on_teach_tj)

dist_teach_on_teach_tj = calculate_distribution(tv_teach_on_teach[1], tv_teach_on_teach[3])
dist_teach_on_opt_tj = calculate_distribution(tv_teach_on_opt[1], tv_teach_on_opt[3])
results['kl_teach_model_trajectory'] = entropy(dist_teach_on_teach_tj, dist_teach_on_opt_tj)

print("\n" + "="*50)
print(" " * 15 + "KL DIVERGENCE RESULTS")
print("="*50)
print("\n## Timestep Divergence ##")
print(f"Optimal Model KL: {results['kl_opt_model_timestep']:.5f}")
print(f"Teaching Model KL: {results['kl_teach_model_timestep']:.5f}")
print("\n## Trajectory Divergence ##")
print(f"Optimal Model KL: {results['kl_opt_model_trajectory']:.5f}")
print(f"Teaching Model KL: {results['kl_teach_model_trajectory']:.5f}")
print("\n" + "="*50)