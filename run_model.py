import numpy as np
import os
import torch
from metadrive.envs.metadrive_env import MetaDriveEnv
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym
from gymnasium import spaces
import time

def create_dummy_env():
    """Create the same dummy environment used during training"""
    class DummyEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Box(
                low=np.array([-np.inf, -np.inf, -np.pi, 0.0], dtype=np.float32),
                high=np.array([np.inf, np.inf, np.pi, np.inf], dtype=np.float32),
                dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=np.array([0, 0, 0, 0], dtype=np.int64),
                high=np.array([1, 1, 1, 1], dtype=np.int64),
                dtype=np.int64
            )
        
        def reset(self, seed=None, options=None):
            return np.zeros(4, dtype=np.float32), {}
        
        def step(self, action):
            return np.zeros(4, dtype=np.float32), 0.0, False, False, {}
    
    return DummyEnv()

def load_bc_model(model_path):
    """Load a BC model saved by the imitation library"""
    try:
        print(f"Attempting to load model from {model_path}")
        
        # Load the saved model data
        model_data = torch.load(model_path, map_location='cpu', weights_only=False)
        print(f"Loaded model data with keys: {list(model_data.keys())}")
        
        # Extract the policy data and state dict
        policy_data = model_data['data']
        state_dict = model_data['state_dict']
        
        # Create dummy environment to get spaces
        dummy_env = create_dummy_env()
        
        # Reconstruct the ActorCriticPolicy
        from stable_baselines3.common.policies import ActorCriticPolicy
        
        # Create policy with the same parameters as during training
        policy = ActorCriticPolicy(
            observation_space=dummy_env.observation_space,
            action_space=dummy_env.action_space,
            lr_schedule=lambda x: 3e-4,  # dummy learning rate schedule
            net_arch=policy_data.get('net_arch', [64, 64]),
            activation_fn=policy_data.get('activation_fn', torch.nn.Tanh),
        )
        
        # Load the state dict
        policy.load_state_dict(state_dict)
        policy.eval()  # Set to evaluation mode
        
        print("Successfully reconstructed BC policy from saved data")
        
        # Create wrapper for predict interface
        class PolicyWrapper:
            def __init__(self, policy):
                self.policy = policy
            
            def predict(self, obs, deterministic=True):
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    # Get actions from the policy
                    actions, _, _ = self.policy(obs_tensor, deterministic=deterministic)
                return actions.cpu().numpy().flatten(), None
        
        return PolicyWrapper(policy)
                
    except Exception as e:
        print(f"Failed to load model {model_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def convert_metadrive_obs_to_model_obs(metadrive_obs, vehicle):
    """Convert MetaDrive observation to our model's 4D observation format"""
    # Get vehicle position, heading, and speed
    position = vehicle.position
    heading = vehicle.heading_theta
    speed = vehicle.speed
    
    # Create 4D observation: [x, y, heading, speed]
    model_obs = np.array([
        position[0],  # x
        position[1],  # y  
        heading,      # heading
        speed         # speed
    ], dtype=np.float32)
    
    return model_obs

def convert_model_action_to_metadrive(model_action):
    """Convert model action to MetaDrive action format"""
    # Model outputs [forward, left, right, brake] (0 or 1)
    # MetaDrive expects [steering, throttle] (-1 to 1)
    
    if len(model_action) >= 4:
        forward, left, right, brake = model_action[:4]
        
        # Convert to steering (-1 to 1)
        steering = (right - left) * 0.5
        
        # Convert to throttle (-1 to 1, negative for brake)
        throttle = forward * 0.8 - brake * 0.5
        
        return np.array([steering, throttle], dtype=np.float32)
    else:
        # If action is already 2D, use as is
        return model_action[:2]

def run_model_in_metadrive(model, map_seed, model_name, max_steps=1000):
    """Run a model in MetaDrive with visual rendering"""
    print(f"\n=== Running {model_name} model in MetaDrive (seed: {map_seed}) ===")
    print("Press ESC or close window to stop the simulation")
    
    # Create MetaDrive environment with no termination conditions
    config = {
        "map": "SCS",
        "traffic_density": 0.1,
        "num_scenarios": 1,
        "start_seed": map_seed,
        "manual_control": False,
        "use_render": True,  # Enable rendering to see trajectories in real-time
        "vehicle_config": {"show_navi_mark": True, "show_line_to_navi_mark": True},
        # Disable all termination conditions to prevent stopping when going out of bounds
        "on_continuous_line_done": False,
        "out_of_route_done": False,
        "out_of_road_done": False,
        "crash_vehicle_done": False,
        "crash_object_done": False,
        "crash_human_done": False,
        "on_broken_line_done": False,
        "horizon": max_steps,  # Set maximum episode length
    }
    
    env = MetaDriveEnv(config)
    
    try:
        # Reset environment
        metadrive_obs, info = env.reset()
        
        print(f"Starting simulation with {model_name} model...")
        print(f"Vehicle starting position: {env.agent.position}")
        
        for step in range(max_steps):
            # Convert MetaDrive observation to model observation
            model_obs = convert_metadrive_obs_to_model_obs(metadrive_obs, env.agent)
            
            # Get action from model
            if model is not None:
                model_action, _ = model.predict(model_obs, deterministic=True)
                # Convert model action to MetaDrive action
                metadrive_action = convert_model_action_to_metadrive(model_action)
            else:
                # Fallback: simple forward action
                metadrive_action = np.array([0.0, 0.5], dtype=np.float32)
            
            # Step environment
            metadrive_obs, reward, terminated, truncated, info = env.step(metadrive_action)
            
            # Print position every 50 steps to show trajectory updates
            if step % 50 == 0:
                pos = env.agent.position
                speed = env.agent.speed
                print(f"Step {step}: Position ({pos[0]:.2f}, {pos[1]:.2f}), Speed: {speed:.2f}")
            
            # Only stop if manually terminated (ESC key or window close)
            # Ignore all other termination conditions
            if terminated and step < 10:  # Only stop if terminated very early (likely manual)
                print(f"Simulation manually stopped at step {step}")
                break
            
            # Small delay to make visualization smoother
            time.sleep(0.02)
        
        print(f"Simulation completed after {step + 1} steps")
        env.close()
        
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
        env.close()
    except Exception as e:
        print(f"Error running model in MetaDrive: {e}")
        env.close()

def main():
    """Main function to run models in MetaDrive with visual rendering"""
    print("Running trained models in MetaDrive with visual rendering...")
    print("Note: All boundary termination conditions are disabled")
    print("The vehicle can drive anywhere without stopping the simulation")
    
    # Map configurations
    maps = [
        {"name": "map_12345", "seed": 12345},
        {"name": "map_54321", "seed": 54321}
    ]
    
    for map_config in maps:
        map_name = map_config["name"]
        map_seed = map_config["seed"]
        
        print(f"\n{'='*60}")
        print(f"Processing {map_name} (seed: {map_seed})")
        print(f"{'='*60}")
        
        # Load models
        optimal_model_path = f"models/optimal_{map_name}_model"
        teaching_model_path = f"models/teaching_{map_name}_model"
        
        print("Loading optimal model...")
        optimal_model = load_bc_model(optimal_model_path)
        
        print("Loading teaching model...")
        teaching_model = load_bc_model(teaching_model_path)
        
        # Run optimal model
        input(f"\nPress Enter to run OPTIMAL model for {map_name}...")
        run_model_in_metadrive(optimal_model, map_seed, "OPTIMAL")
        
        # Run teaching model
        input(f"\nPress Enter to run TEACHING model for {map_name}...")
        run_model_in_metadrive(teaching_model, map_seed, "TEACHING")
    
    print("\nAll model demonstrations complete!")

if __name__ == "__main__":
    main()
