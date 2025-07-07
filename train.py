import numpy as np
import gymnasium as gym
import sys
import os
import glob
from datetime import datetime

from imitation.algorithms import bc
from imitation.data import types, serialize
from imitation.data.rollout import flatten_trajectories
from imitation.util.networks import RunningNorm
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
import torch

# Training parameters
batch_size = 32
epochs = 100
learning_rate = 3e-4

def load_trajectories_from_directory(directory_path):
    """Load trajectories from a directory using imitation library's serialize.load()"""
    trajectories = []
    
    if not os.path.exists(directory_path):
        return trajectories
    
    # Find all trajectory files in the directory
    trajectory_files = glob.glob(os.path.join(directory_path, "trajectory_*"))
    
    for file_path in trajectory_files:
        try:
            # Load trajectory using imitation library
            loaded_trajectories = serialize.load(file_path)
            trajectories.extend(loaded_trajectories)
            print(f"Loaded trajectory from {file_path} with {len(loaded_trajectories)} trajectories")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return trajectories

def extract_data():
    """Extract trajectory data from the trajectory_data directory structure"""
    maps_data = {}
    
    if not os.path.exists("trajectory_data"):
        print("trajectory_data directory not found!")
        return maps_data
    
    # Load from map_* folders
    map_paths = glob.glob(os.path.join("trajectory_data", "map_*"))
    for map_path in map_paths:
        map_name = os.path.basename(map_path)
        maps_data[map_name] = {'optimal': [], 'teaching': []}
        
        # Load optimal trajectories
        optimal_dir = os.path.join(map_path, "optimal")
        optimal_trajectories = load_trajectories_from_directory(optimal_dir)
        maps_data[map_name]['optimal'] = optimal_trajectories
        
        # Load teaching trajectories
        teaching_dir = os.path.join(map_path, "teaching")
        teaching_trajectories = load_trajectories_from_directory(teaching_dir)
        maps_data[map_name]['teaching'] = teaching_trajectories
        
        print(f"Map {map_name}: {len(optimal_trajectories)} optimal, {len(teaching_trajectories)} teaching trajectories")
    
    return maps_data

def create_environment():
    """Create a custom environment that matches the trajectory data format"""
    import gymnasium as gym
    from gymnasium import spaces
    
    # Create a dummy environment that matches our trajectory data format
    # Our trajectory data has 4-dimensional observations: [x, y, heading, speed]
    # and 4-dimensional actions: [forward, left, right, brake]
    
    class DummyEnv(gym.Env):
        def __init__(self):
            super().__init__()
            # Match the observation space to our trajectory data (4 dimensions)
            self.observation_space = spaces.Box(
                low=np.array([-np.inf, -np.inf, -np.pi, 0.0], dtype=np.float32),
                high=np.array([np.inf, np.inf, np.pi, np.inf], dtype=np.float32),
                dtype=np.float32
            )
            # Match the action space to our trajectory data (4 discrete actions)
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
def train_imitation_model(trajectories, model_name, demo_type):
    """Train an imitation learning model using the given trajectories"""
    if not trajectories:
        print(f"No {demo_type} trajectories found for {model_name}")
        return None
    
    print(f"Training {demo_type} model for {model_name} with {len(trajectories)} trajectories")
    
    # Create environment to get observation and action spaces
    env = create_environment()
    obs_space = env.observation_space
    act_space = env.action_space
    transitions = flatten_trajectories(trajectories)
    
    print(f"Converted to {len(transitions)} total transitions.")
    
    # Create BC trainer
    bc_trainer = bc.BC(
        observation_space=obs_space,
        action_space=act_space,
        demonstrations=transitions, # Pass the correctly formatted transitions
        rng=np.random.default_rng(0),
        batch_size=batch_size,
        optimizer_cls=torch.optim.Adam,
        custom_logger=None,
    )
    
    # Train the model
    print(f"Starting training for {epochs} epochs...")
    bc_trainer.train(n_epochs=epochs)
    
    # Save the trained model
    model_save_path = f"models/{demo_type}_{model_name}_model"
    os.makedirs("models", exist_ok=True)
    bc_trainer.policy.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    env.close()
    return bc_trainer

def main():
    print("Loading trajectory data...")
    maps_data = extract_data()
    
    if not maps_data:
        print("No trajectory data found!")
        sys.exit(1)
    
    print(f"Found data for {len(maps_data)} maps")
    
    # Train models for each map and demonstration type
    for map_name, trajectories in maps_data.items():
        print(f"\n=== Training models for {map_name} ===")
        
        # Train optimal model
        if trajectories['optimal']:
            train_imitation_model(trajectories['optimal'], map_name, "optimal")
        else:
            print(f"No optimal trajectories found for {map_name}")
        
        # Train teaching model
        if trajectories['teaching']:
            train_imitation_model(trajectories['teaching'], map_name, "teaching")
        else:
            print(f"No teaching trajectories found for {map_name}")
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main()
