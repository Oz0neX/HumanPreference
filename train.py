import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import glob
from typing import List, Dict
from sklearn.preprocessing import StandardScaler
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrajectoryDataset(Dataset):
    def __init__(self, states: np.ndarray, actions: np.ndarray):
        self.states = torch.tensor(states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.float32)

    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

class TrajectoryLoader:
    def __init__(self, data_dir: str = "trajectory_data"):
        self.data_dir = data_dir
        self.state_cols = ['vehicle_x', 'vehicle_y', 'vehicle_heading', 'vehicle_speed']
        self.action_cols = ['forward', 'left', 'right', 'brake']

    def load_and_group_trajectories(self) -> Dict[str, List[Dict]]:
        trajectories = {'optimal': [], 'teaching': []}
        if not os.path.exists(self.data_dir):
            print(f"Data directory {self.data_dir} not found.")
            return trajectories
        
        # Load from both optimal_session_* and teaching_session_* folders
        session_patterns = ["optimal_session_*", "teaching_session_*"]
        for pattern in session_patterns:
            session_paths = glob.glob(os.path.join(self.data_dir, pattern))
            for session_path in session_paths:
                session_name = os.path.basename(session_path)
                print(f"Processing session: {session_name}")
                for group in trajectories.keys():
                    group_dir = os.path.join(session_path, group)
                    if not os.path.exists(group_dir): continue
                    for file_path in glob.glob(os.path.join(group_dir, "*.csv")):
                        try:
                            df = pd.read_csv(file_path)
                            print(f"Loading {file_path}")
                            trajectories[group].append({
                                'states': df[self.state_cols].values,
                                'actions': df[self.action_cols].values
                            })
                        except Exception as e:
                            print(f"Error loading {file_path}: {e}")
        return trajectories

class ImitationLearningModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 64]):
        super().__init__()
        self.lay1 = nn.Linear(state_dim, hidden_dims[0])
        self.lay2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.classify = nn.Linear(hidden_dims[1], action_dim)

    def forward(self, states):
        out = torch.relu(self.lay1(states))
        out = torch.relu(self.lay2(out))
        out = self.classify(out)
        return torch.sigmoid(out)  # Add sigmoid to ensure output is between 0 and 1

class ImitationLearningTrainer:
    def __init__(self, model: nn.Module, learning_rate: float = 0.001):
        self.model = model.to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        self.train_losses = []
        self.val_losses = []

    def _run_epoch(self, loader: DataLoader, is_training: bool) -> float:
        self.model.train(is_training)
        total_loss = 0.0
        if is_training:
            torch.enable_grad()
        else:
            torch.no_grad()
        for states, actions in loader:
            states, actions = states.to(device), actions.to(device)
            if is_training: self.optimizer.zero_grad()
            predictions = self.model(states)
            loss = self.criterion(predictions, actions)
            if is_training:
                loss.backward()
                self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 100):
        print(f"Training for {epochs} epochs...")
        for epoch in range(epochs):
            train_loss = self._run_epoch(train_loader, is_training=True)
            val_loss = self._run_epoch(val_loader, is_training=False)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        print("Training completed!")

    def save_model(self, path: str):
        torch.save({'model_state_dict': self.model.state_dict()}, path)
        print(f"Model saved to {path}")

def train_model_on_trajectories(trajectories: List[Dict], model_name: str, timestamp: str):
    if not trajectories:
        print(f"No {model_name} trajectories available for training.")
        return
    
    print(f"\n--- Training {model_name.capitalize()} Model ---")
    
    all_states = np.vstack([traj['states'] for traj in trajectories])
    all_actions = np.vstack([traj['actions'] for traj in trajectories])
    
    # Validate and clean action data
    print(f"Action data shape: {all_actions.shape}")
    print(f"Action data range: min={np.min(all_actions)}, max={np.max(all_actions)}")
    print(f"Action data contains NaN: {np.isnan(all_actions).any()}")
    print(f"Action data contains Inf: {np.isinf(all_actions).any()}")
    
    # Ensure actions are binary (0 or 1) and handle any invalid values
    all_actions = np.clip(all_actions, 0, 1)  # Clip to [0,1] range
    all_actions = np.nan_to_num(all_actions, nan=0.0, posinf=1.0, neginf=0.0)  # Handle NaN/Inf
    
    print(f"After cleaning - Action range: min={np.min(all_actions)}, max={np.max(all_actions)}")
    
    scaler = StandardScaler()
    scaled_states = scaler.fit_transform(all_states)

    dataset = TrajectoryDataset(scaled_states, all_actions)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = ImitationLearningModel(state_dim=4, action_dim=4, hidden_dims=[128, 64, 32])
    trainer = ImitationLearningTrainer(model, learning_rate=0.001)
    trainer.train(train_loader, val_loader, epochs=100)
    
    trainer.save_model(f"il_model_{model_name}_{timestamp}.pth")

def main():
    print("Start train")
    loader = TrajectoryLoader()
    trajectory_matrix = loader.load_and_group_trajectories()
    
    if not trajectory_matrix['optimal'] and not trajectory_matrix['teaching']:
        print("No trajectory data found.")
        return
        
    print(f"\nFound {len(trajectory_matrix['optimal'])} optimal and {len(trajectory_matrix['teaching'])} teaching trajectories.")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_model_on_trajectories(trajectory_matrix['optimal'], "optimal", timestamp)
    train_model_on_trajectories(trajectory_matrix['teaching'], "teaching", timestamp)
    
    print(f"Saved models")

if __name__ == "__main__":
    main()
