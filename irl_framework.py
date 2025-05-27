"""
IRL Framework for iterative feedback experiments.
Handles trajectory collection, IRL learning, and policy updates.
"""

import numpy as np
import json
import csv
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pickle


@dataclass
class TrajectoryData:
    """Container for trajectory information."""
    states: List[np.ndarray]
    actions: List[np.ndarray]
    positions: List[Tuple[float, float]]
    episode_id: int
    agent_type: str  # 'human' or 'agent'
    timestamp: str
    metadata: Dict = None


class TrajectoryManager:
    """Manages trajectory collection and storage."""
    
    def __init__(self, base_log_dir: str):
        self.base_log_dir = base_log_dir
        self.current_trajectory = None
        self.all_trajectories = []
        
    def start_new_trajectory(self, episode_id: int, agent_type: str) -> None:
        """Start recording a new trajectory."""
        self.current_trajectory = TrajectoryData(
            states=[],
            actions=[],
            positions=[],
            episode_id=episode_id,
            agent_type=agent_type,
            timestamp=datetime.now().isoformat(),
            metadata={}
        )
        
    def add_step(self, state: np.ndarray, action: np.ndarray, position: Tuple[float, float]) -> None:
        """Add a step to the current trajectory."""
        if self.current_trajectory is not None:
            self.current_trajectory.states.append(state.copy())
            self.current_trajectory.actions.append(action.copy())
            self.current_trajectory.positions.append(position)
            
    def finish_trajectory(self) -> TrajectoryData:
        """Finish the current trajectory and return it."""
        if self.current_trajectory is not None:
            trajectory = self.current_trajectory
            self.all_trajectories.append(trajectory)
            self.current_trajectory = None
            return trajectory
        return None
        
    def save_trajectory(self, trajectory: TrajectoryData, save_format: str = 'both') -> None:
        """Save trajectory to disk in specified format."""
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        if save_format in ['csv', 'both']:
            # Save as CSV (compatible with existing system)
            csv_path = os.path.join(
                self.base_log_dir, 
                f"{trajectory.agent_type}_episode_{trajectory.episode_id}_{timestamp_str}.csv"
            )
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["step", "pos_x", "pos_y", "action_0", "action_1"])
                for i, (pos, action) in enumerate(zip(trajectory.positions, trajectory.actions)):
                    writer.writerow([i, pos[0], pos[1], action[0], action[1]])
                    
        if save_format in ['pickle', 'both']:
            # Save as pickle (preserves full state information)
            pickle_path = os.path.join(
                self.base_log_dir,
                f"{trajectory.agent_type}_episode_{trajectory.episode_id}_{timestamp_str}.pkl"
            )
            with open(pickle_path, 'wb') as f:
                pickle.dump(trajectory, f)
                
    def get_human_trajectories(self) -> List[TrajectoryData]:
        """Get all human trajectories."""
        return [t for t in self.all_trajectories if t.agent_type == 'human']
        
    def get_agent_trajectories(self) -> List[TrajectoryData]:
        """Get all agent trajectories."""
        return [t for t in self.all_trajectories if t.agent_type == 'agent']


class SimpleIRLLearner:
    """
    A simple IRL implementation for demonstration.
    This is a placeholder that can be replaced with more sophisticated IRL algorithms.
    """
    
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_weights = np.random.normal(0, 0.1, size=(state_dim,))
        self.learning_rate = 0.01
        self.iteration = 0
        
    def compute_reward(self, state: np.ndarray) -> float:
        """Compute reward for a given state using learned weights."""
        return np.dot(state, self.reward_weights)
        
    def update_from_trajectories(self, human_trajectories: List[TrajectoryData], 
                                agent_trajectories: List[TrajectoryData] = None) -> Dict:
        """
        Update reward function based on human demonstrations.
        This is a simplified version - real IRL would be more sophisticated.
        """
        if not human_trajectories:
            return {"status": "no_trajectories", "iteration": self.iteration}
            
        # Simple approach: adjust weights to prefer states visited by humans
        all_human_states = []
        for traj in human_trajectories:
            all_human_states.extend(traj.states)
            
        if not all_human_states:
            return {"status": "no_states", "iteration": self.iteration}
            
        # Convert to numpy array
        human_states = np.array(all_human_states)
        
        # Simple gradient ascent to increase reward for human-visited states
        if len(human_states) > 0:
            # Compute mean state features from human demonstrations
            mean_human_features = np.mean(human_states, axis=0)
            
            # Update weights to increase reward for these features
            self.reward_weights += self.learning_rate * mean_human_features
            
            # Normalize to prevent unbounded growth
            self.reward_weights = self.reward_weights / (np.linalg.norm(self.reward_weights) + 1e-8)
            
        self.iteration += 1
        
        return {
            "status": "updated",
            "iteration": self.iteration,
            "reward_weights": self.reward_weights.copy(),
            "num_human_trajectories": len(human_trajectories),
            "num_human_states": len(all_human_states)
        }
        
    def get_policy_weights(self) -> np.ndarray:
        """Get current policy weights (placeholder for actual policy learning)."""
        # This is a placeholder - in real IRL, you'd train a policy using the learned reward
        return self.reward_weights.copy()


class IterativeFeedbackExperiment:
    """
    Main controller for the iterative feedback experiment.
    Manages the alternating human-agent demonstration process.
    """
    
    def __init__(self, experiment_name: str, base_log_dir: str):
        self.experiment_name = experiment_name
        self.base_log_dir = base_log_dir
        self.experiment_dir = os.path.join(base_log_dir, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        self.trajectory_manager = TrajectoryManager(self.experiment_dir)
        self.irl_learner = None  # Will be initialized when we know state dimensions
        
        self.current_phase = "human"  # "human" or "agent"
        self.iteration = 0
        self.episode_counter = 0
        
        # Experiment state
        self.experiment_log = []
        
    def initialize_irl(self, state_dim: int, action_dim: int = 2):
        """Initialize the IRL learner with known dimensions."""
        self.irl_learner = SimpleIRLLearner(state_dim, action_dim)
        print(f"IRL learner initialized with state_dim={state_dim}, action_dim={action_dim}")
        
    def start_human_phase(self) -> int:
        """Start a human demonstration phase."""
        self.current_phase = "human"
        self.episode_counter += 1
        episode_id = self.episode_counter
        
        self.trajectory_manager.start_new_trajectory(episode_id, "human")
        
        print(f"\n=== HUMAN DEMONSTRATION PHASE ===")
        print(f"Iteration: {self.iteration}")
        print(f"Episode: {episode_id}")
        print("Please drive the vehicle. Your trajectory will be recorded.")
        
        return episode_id
        
    def finish_human_phase(self) -> TrajectoryData:
        """Finish human demonstration and save trajectory."""
        trajectory = self.trajectory_manager.finish_trajectory()
        if trajectory:
            self.trajectory_manager.save_trajectory(trajectory)
            print(f"Human trajectory saved: {len(trajectory.positions)} steps")
            
        return trajectory
        
    def start_agent_phase(self) -> int:
        """Start an agent demonstration phase."""
        self.current_phase = "agent"
        self.episode_counter += 1
        episode_id = self.episode_counter
        
        self.trajectory_manager.start_new_trajectory(episode_id, "agent")
        
        print(f"\n=== AGENT DEMONSTRATION PHASE ===")
        print(f"Iteration: {self.iteration}")
        print(f"Episode: {episode_id}")
        print("Agent will now demonstrate what it has learned.")
        
        return episode_id
        
    def finish_agent_phase(self) -> TrajectoryData:
        """Finish agent demonstration and save trajectory."""
        trajectory = self.trajectory_manager.finish_trajectory()
        if trajectory:
            self.trajectory_manager.save_trajectory(trajectory)
            print(f"Agent trajectory saved: {len(trajectory.positions)} steps")
            
        return trajectory
        
    def update_irl_model(self) -> Dict:
        """Update the IRL model with collected human trajectories."""
        if self.irl_learner is None:
            return {"status": "irl_not_initialized"}
            
        human_trajectories = self.trajectory_manager.get_human_trajectories()
        agent_trajectories = self.trajectory_manager.get_agent_trajectories()
        
        update_result = self.irl_learner.update_from_trajectories(
            human_trajectories, agent_trajectories
        )
        
        print(f"\n=== IRL MODEL UPDATE ===")
        print(f"Status: {update_result['status']}")
        print(f"IRL Iteration: {update_result['iteration']}")
        if 'num_human_trajectories' in update_result:
            print(f"Human trajectories used: {update_result['num_human_trajectories']}")
            
        # Log the update
        self.experiment_log.append({
            "timestamp": datetime.now().isoformat(),
            "iteration": self.iteration,
            "phase": "irl_update",
            "update_result": update_result
        })
        
        return update_result
        
    def get_updated_policy_weights(self) -> Optional[np.ndarray]:
        """Get updated policy weights from IRL learning."""
        if self.irl_learner is None:
            return None
        return self.irl_learner.get_policy_weights()
        
    def next_iteration(self):
        """Move to the next iteration of the experiment."""
        self.iteration += 1
        print(f"\n=== STARTING ITERATION {self.iteration} ===")
        
    def record_step(self, state: np.ndarray, action: np.ndarray, position: Tuple[float, float]):
        """Record a step in the current trajectory."""
        self.trajectory_manager.add_step(state, action, position)
        
        # Initialize IRL learner if this is the first state we see
        if self.irl_learner is None and len(state) > 0:
            self.initialize_irl(len(state))
            
    def save_experiment_state(self):
        """Save the current experiment state."""
        state_file = os.path.join(self.experiment_dir, "experiment_state.json")
        
        # Convert experiment log to JSON-serializable format
        serializable_log = []
        for entry in self.experiment_log:
            serializable_entry = {}
            for key, value in entry.items():
                if isinstance(value, dict) and "reward_weights" in value:
                    # Convert numpy arrays to lists for JSON serialization
                    serializable_value = value.copy()
                    if isinstance(serializable_value.get("reward_weights"), np.ndarray):
                        serializable_value["reward_weights"] = serializable_value["reward_weights"].tolist()
                    serializable_entry[key] = serializable_value
                else:
                    serializable_entry[key] = value
            serializable_log.append(serializable_entry)
        
        state_data = {
            "experiment_name": self.experiment_name,
            "iteration": self.iteration,
            "episode_counter": self.episode_counter,
            "current_phase": self.current_phase,
            "experiment_log": serializable_log
        }
        
        with open(state_file, 'w') as f:
            json.dump(state_data, f, indent=2)
            
    def get_experiment_summary(self) -> Dict:
        """Get a summary of the experiment progress."""
        human_trajs = self.trajectory_manager.get_human_trajectories()
        agent_trajs = self.trajectory_manager.get_agent_trajectories()
        
        return {
            "experiment_name": self.experiment_name,
            "iteration": self.iteration,
            "total_episodes": self.episode_counter,
            "human_trajectories": len(human_trajs),
            "agent_trajectories": len(agent_trajs),
            "current_phase": self.current_phase,
            "irl_initialized": self.irl_learner is not None
        }
