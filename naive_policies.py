"""
Naive driving policies for IRL experiments.
These policies represent agents that don't know how to drive well initially.
"""

import numpy as np
from metadrive.policy.base_policy import BasePolicy
from metadrive.examples.ppo_expert.numpy_expert import expert


class RandomPolicy(BasePolicy):
    """
    A completely random driving policy.
    Outputs random steering and throttle/brake actions.
    """
    DEBUG_MARK_COLOR = (255, 0, 0, 255)  # Red mark
    
    def __init__(self, control_object, random_seed=None, config=None):
        super().__init__(control_object, random_seed, config)
        self.action_range = 1.0
        
    def act(self, *args, **kwargs):
        # Random steering [-1, 1] and throttle/brake [-1, 1]
        steering = self.np_random.uniform(-self.action_range, self.action_range)
        throttle_brake = self.np_random.uniform(-self.action_range, self.action_range)
        
        action = np.array([steering, throttle_brake], dtype=np.float32)
        self.action_info["policy_type"] = "random"
        self.action_info["action"] = action
        return action


class NoisyExpertPolicy(BasePolicy):
    """
    Expert policy with added noise to simulate poor driving.
    Uses the existing expert but degrades performance with noise.
    """
    DEBUG_MARK_COLOR = (255, 165, 0, 255)  # Orange mark
    
    def __init__(self, control_object, random_seed=None, config=None):
        super().__init__(control_object, random_seed, config)
        # Noise parameters - can be adjusted to control "badness"
        self.steering_noise_std = 0.3
        self.throttle_noise_std = 0.2
        self.action_corruption_prob = 0.1  # Probability of completely random action
        
    def act(self, *args, **kwargs):
        try:
            # Get expert action
            expert_action = expert(self.control_object, deterministic=True)
            
            # Add noise to expert action
            if self.np_random.random() < self.action_corruption_prob:
                # Sometimes completely ignore expert and do random action
                action = np.array([
                    self.np_random.uniform(-1, 1),
                    self.np_random.uniform(-1, 1)
                ], dtype=np.float32)
            else:
                # Add gaussian noise to expert action
                steering_noise = self.np_random.normal(0, self.steering_noise_std)
                throttle_noise = self.np_random.normal(0, self.throttle_noise_std)
                
                action = expert_action + np.array([steering_noise, throttle_noise])
                action = np.clip(action, -1.0, 1.0)
            
            self.action_info["policy_type"] = "noisy_expert"
            self.action_info["action"] = action
            return action
            
        except Exception as e:
            # Fallback to random if expert fails
            print(f"Expert failed, using random action: {e}")
            return np.array([
                self.np_random.uniform(-1, 1),
                self.np_random.uniform(-1, 1)
            ], dtype=np.float32)


class SimpleForwardPolicy(BasePolicy):
    """
    A simple policy that tries to go forward but has poor control.
    Represents a very basic driving attempt.
    """
    DEBUG_MARK_COLOR = (0, 255, 0, 255)  # Green mark
    
    def __init__(self, control_object, random_seed=None, config=None):
        super().__init__(control_object, random_seed, config)
        self.target_speed = 0.5  # Moderate speed
        self.steering_sensitivity = 0.1
        
    def act(self, *args, **kwargs):
        # Simple forward driving with basic steering correction
        vehicle = self.control_object
        
        # Get basic vehicle state
        velocity = vehicle.velocity
        current_speed = np.linalg.norm([velocity[0], velocity[1]])
        
        # Simple throttle control - try to maintain target speed
        if current_speed < self.target_speed:
            throttle = 0.5
        else:
            throttle = 0.1
            
        # Very basic steering - add some randomness to simulate poor control
        base_steering = self.np_random.uniform(-0.2, 0.2)
        
        # Try to stay somewhat straight but with poor control
        if hasattr(vehicle, 'heading_theta'):
            # Add small correction based on heading, but with noise
            heading_correction = self.np_random.normal(0, 0.1)
            steering = base_steering + heading_correction
        else:
            steering = base_steering
            
        steering = np.clip(steering, -1.0, 1.0)
        
        action = np.array([steering, throttle], dtype=np.float32)
        self.action_info["policy_type"] = "simple_forward"
        self.action_info["action"] = action
        return action


class ProgressivePolicy(BasePolicy):
    """
    A policy that can be updated with new parameters/weights.
    This will be used for the IRL-learned policy that improves over time.
    """
    DEBUG_MARK_COLOR = (0, 0, 255, 255)  # Blue mark
    
    def __init__(self, control_object, random_seed=None, config=None):
        super().__init__(control_object, random_seed, config)
        self.policy_weights = None
        self.reward_function = None
        self.fallback_policy = SimpleForwardPolicy(control_object, random_seed, config)
        self.learning_iteration = 0
        
    def update_policy(self, new_weights=None, new_reward_function=None):
        """Update the policy with new learned parameters."""
        if new_weights is not None:
            self.policy_weights = new_weights
        if new_reward_function is not None:
            self.reward_function = new_reward_function
        self.learning_iteration += 1
        print(f"Policy updated - Iteration {self.learning_iteration}")
        
    def act(self, *args, **kwargs):
        # If no learned policy yet, use fallback
        if self.policy_weights is None:
            action = self.fallback_policy.act(*args, **kwargs)
            self.action_info["policy_type"] = "progressive_fallback"
        else:
            # Use learned policy (placeholder for now - will integrate with actual IRL)
            # For now, interpolate between fallback and expert based on learning progress
            fallback_action = self.fallback_policy.act(*args, **kwargs)
            
            try:
                expert_action = expert(self.control_object, deterministic=True)
                # Linear interpolation based on learning iteration
                alpha = min(self.learning_iteration * 0.1, 1.0)  # Gradually improve
                action = (1 - alpha) * fallback_action + alpha * expert_action
                action = np.clip(action, -1.0, 1.0)
                self.action_info["policy_type"] = "progressive_learned"
            except:
                action = fallback_action
                self.action_info["policy_type"] = "progressive_fallback"
        
        self.action_info["learning_iteration"] = self.learning_iteration
        self.action_info["action"] = action
        return action


def get_naive_policy(policy_type, control_object, random_seed=None, config=None):
    """
    Factory function to create naive policies.
    
    Args:
        policy_type: str, one of ['random', 'noisy_expert', 'simple_forward', 'progressive']
        control_object: The vehicle to control
        random_seed: Random seed for reproducibility
        config: Policy configuration
        
    Returns:
        BasePolicy instance
    """
    policy_map = {
        'random': RandomPolicy,
        'noisy_expert': NoisyExpertPolicy,
        'simple_forward': SimpleForwardPolicy,
        'progressive': ProgressivePolicy
    }
    
    if policy_type not in policy_map:
        raise ValueError(f"Unknown policy type: {policy_type}. Available: {list(policy_map.keys())}")
    
    return policy_map[policy_type](control_object, random_seed, config)
