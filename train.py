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
    
    obs = env.reset()[0]
    action_distributions = []
    action_grid = torch.linspace(-1, 1, 100)
    
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
                'steering_probs': steering_probs,
                'throttle_probs': throttle_probs
            })
            
            action = bc_trainer.policy.action_dist.sample()
        
        obs, _, terminated, truncated, _ = env.step(action.cpu().numpy().flatten())
        if terminated or truncated:
            break
    
    print(f"Captured {len(action_distributions)} timesteps of action distributions")
    env.close()
    
    STEPS = 1000
    num_rollouts = 10
    all_trajectories_x = []
    all_trajectories_y = []
    
    for rollout in range(num_rollouts):
        obs = env.reset()[0]
        trajectory_x = []
        trajectory_y = []
        
        for i in range(min(len(action_distributions), STEPS)):  
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                features = bc_trainer.policy.extract_features(obs_tensor)
                latent_pi = bc_trainer.policy.mlp_extractor.forward_actor(features)
                mean_actions = bc_trainer.policy.action_net(latent_pi)
                log_std = bc_trainer.policy.log_std
                bc_trainer.policy.action_dist.proba_distribution(mean_actions, log_std)
                action = bc_trainer.policy.action_dist.sample()
            
            trajectory_x.append(env.agent.position[0])
            trajectory_y.append(env.agent.position[1])
            obs, _, terminated, truncated, _ = env.step(action.cpu().numpy().flatten())
            if terminated or truncated:
                break
        
        all_trajectories_x.append(trajectory_x)
        all_trajectories_y.append(trajectory_y)
    
    min_length = min(len(traj) for traj in all_trajectories_x)
    trajectory_x_array = np.array([traj[:min_length] for traj in all_trajectories_x])
    trajectory_y_array = np.array([traj[:min_length] for traj in all_trajectories_y])
    
    mean_x = np.mean(trajectory_x_array, axis=0)
    mean_y = np.mean(trajectory_y_array, axis=0)
    std_x = np.std(trajectory_x_array, axis=0)
    std_y = np.std(trajectory_y_array, axis=0)
    
    timesteps_to_show = min(STEPS, len(trajectory_x))
    steering_heatmap = np.array([dist['steering_probs'] for dist in action_distributions[:timesteps_to_show]])
    throttle_heatmap = np.array([dist['throttle_probs'] for dist in action_distributions[:timesteps_to_show]])
    
    steering_scaler = MinMaxScaler(feature_range=(-1, 1))
    throttle_scaler = MinMaxScaler(feature_range=(-1, 1))
    
    steering_heatmap = steering_scaler.fit_transform(steering_heatmap.flatten().reshape(-1, 1)).reshape(steering_heatmap.shape)
    throttle_heatmap = throttle_scaler.fit_transform(throttle_heatmap.flatten().reshape(-1, 1)).reshape(throttle_heatmap.shape)
    
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    timesteps_to_show = min(min_length, STEPS)
    
    # Plot +2σ and -2σ trajectories (outermost - yellow)
    plt.plot(mean_y[:timesteps_to_show] + 2*std_y[:timesteps_to_show], 
             mean_x[:timesteps_to_show] + 2*std_x[:timesteps_to_show], 
             'yellow', alpha=0.7, linewidth=2)
    plt.plot(mean_y[:timesteps_to_show] - 2*std_y[:timesteps_to_show], 
             mean_x[:timesteps_to_show] - 2*std_x[:timesteps_to_show], 
             'yellow', alpha=0.7, linewidth=2)
    
    # Plot +1σ and -1σ trajectories (middle - orange)
    plt.plot(mean_y[:timesteps_to_show] + std_y[:timesteps_to_show], 
             mean_x[:timesteps_to_show] + std_x[:timesteps_to_show], 
             'orange', alpha=0.7, linewidth=2)
    plt.plot(mean_y[:timesteps_to_show] - std_y[:timesteps_to_show], 
             mean_x[:timesteps_to_show] - std_x[:timesteps_to_show], 
             'orange', alpha=0.7, linewidth=2)
    plt.plot(mean_y[:timesteps_to_show], mean_x[:timesteps_to_show], 'red', linewidth=3)
    plt.scatter(mean_y[0], mean_x[0], color='green', s=100, label='Start')
    plt.scatter(mean_y[timesteps_to_show-1], mean_x[timesteps_to_show-1], color='red', s=100, label='End')
    
    import matplotlib.cm as cm
    colors = cm.rainbow(np.linspace(0, 1, len(range(10, timesteps_to_show, 10))))
    for idx, i in enumerate(range(10, timesteps_to_show, 10)):
        plt.scatter(mean_y[i], mean_x[i], color=colors[idx], s=75)
    
    plt.title('Vehicle Trajectory with Uncertainty')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.gca().invert_xaxis()
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.imshow(np.fliplr(steering_heatmap), cmap='coolwarm', aspect='auto', vmin=0, vmax=1, origin='lower')
    plt.colorbar(label='Probability')
    plt.title('Steering Probability Heatmap')
    plt.xlabel('[Left] <-- Steering Direction --> [Right]')
    plt.ylabel('Timestep')
    for idx, i in enumerate(range(10, timesteps_to_show, 10)):
        plt.axhline(y=i, color=colors[idx], linewidth=2)
    
    plt.subplot(1, 3, 3)
    plt.imshow(throttle_heatmap, cmap='coolwarm', aspect='auto', vmin=0, vmax=1, origin='lower')
    plt.colorbar(label='Probability')
    plt.title('Throttle Probability Heatmap')
    plt.xlabel('[Break] <-- Acceleration --> [Throttle]')
    plt.ylabel('Timestep')
    for idx, i in enumerate(range(10, timesteps_to_show, 10)):
        plt.axhline(y=i, color=colors[idx], linewidth=2)
    
    plt.tight_layout()
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
