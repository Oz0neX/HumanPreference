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

    custom_policy = ActorCriticPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: 3e-4,
        net_arch=[32, 128, 128, 32]
    )
    
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=np.random.default_rng(0),
        policy=custom_policy,
        batch_size=50,
    )

    bc_trainer.train(n_epochs=100)
    
    obs = env.reset()[0]
    action_distributions = []
    action_grid = torch.linspace(-1, 1, 21)
    
    for _ in range(1000):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            features = bc_trainer.policy.extract_features(obs_tensor)
            latent_pi = bc_trainer.policy.mlp_extractor.forward_actor(features)
            mean_actions = bc_trainer.policy.action_net(latent_pi)
            log_std = bc_trainer.policy.log_std
            
            bc_trainer.policy.action_dist.proba_distribution(mean_actions, log_std)
            
            steering_logits = []
            throttle_logits = []
            
            for steering_val in action_grid:
                test_action = torch.tensor([[steering_val, action_grid[0]]])
                log_prob = bc_trainer.policy.action_dist.log_prob(test_action)
                steering_logits.append(log_prob.item())
                
            for throttle_val in action_grid:
                test_action = torch.tensor([[action_grid[0], throttle_val]])
                log_prob = bc_trainer.policy.action_dist.log_prob(test_action)
                throttle_logits.append(log_prob.item())
            
            steering_logits = torch.tensor(steering_logits)
            throttle_logits = torch.tensor(throttle_logits)
            steering_probs = F.softmax(steering_logits, dim=0).numpy()
            throttle_probs = F.softmax(throttle_logits, dim=0).numpy()
            
            action_distributions.append({
                'steering_grid': action_grid.numpy(),
                'steering_probs': steering_probs,
                'throttle_grid': action_grid.numpy(), 
                'throttle_probs': throttle_probs
            })
            
            action = bc_trainer.policy.action_dist.sample()
        
        obs, _, terminated, truncated, _ = env.step(action.cpu().numpy().flatten())
        if terminated or truncated:
            break
    
    print(f"Captured {len(action_distributions)} timesteps of action distributions")
    env.close()
    
    print(np.array(action_distributions).shape)
    timesteps_to_show = min(50, len(action_distributions))
    
    plt.figure(figsize=(50, 40))
    
    for i in range(timesteps_to_show):
        plt.subplot(10, 5, i+1)
        plt.plot(action_distributions[i]['steering_grid'], action_distributions[i]['steering_probs'], label='Steering', color='blue')
        plt.plot(action_distributions[i]['throttle_grid'], action_distributions[i]['throttle_probs'], label='Throttle', color='red')
        plt.title(f'Timestep {i + 1}')
        plt.xlabel('Action Value')
        plt.ylabel('Probability')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'action_distributions_seed_{seed}.png')
    plt.show()
    sys.exit()
    
for map_name, trajectories_dict in maps_data.items():
    seed = int(map_name.split('_')[1])
    
    optimal_trajectories = trajectories_dict.get('optimal', [])
    optimal_transitions = flatten_trajectories(optimal_trajectories)
    train(optimal_transitions, seed)
    
    teaching_trajectories = trajectories_dict.get('teaching', [])
    teaching_transitions = flatten_trajectories(teaching_trajectories)
    train(teaching_transitions, seed)
