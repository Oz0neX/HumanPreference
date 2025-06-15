#!/usr/bin/env python

import numpy as np
from metadrive.envs.top_down_env import TopDownMetaDrive
from metadrive.examples.ppo_expert.numpy_expert import expert

class NoisyExpertPolicy:
    
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.steering_noise = 0.3
        self.throttle_noise = 0.2
        self.corruption_prob = 0.15
        
    def act(self):
        try:
            expert_action = expert(self.vehicle, deterministic=True)
            
            if np.random.random() < self.corruption_prob:
                action = np.array([
                    np.random.uniform(-1, 1),
                    np.random.uniform(-1, 1)
                ])
            else:
                steering_noise = np.random.normal(0, self.steering_noise)
                throttle_noise = np.random.normal(0, self.throttle_noise)
                action = expert_action + np.array([steering_noise, throttle_noise])
                action = np.clip(action, -1.0, 1.0)
            
            return action
        except:
            return np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])


class NaiveIRLPolicy:
    
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.iteration = 0
        
    def act(self):
        # Implement naive IRL policy here
        if self.iteration == 0:
            steering = np.random.uniform(-0.8, 0.8)
            throttle = np.random.uniform(0.1, 0.6)
            return np.array([steering, throttle])
            
        elif self.iteration == 1:
            steering = np.random.uniform(-0.4, 0.4)
            throttle = np.random.uniform(0.2, 0.7)
            return np.array([steering, throttle])
            
        else:
            steering = np.random.uniform(-0.2, 0.2)
            throttle = 0.5
            return np.array([steering, throttle])


def run_demonstration(policy_type, iteration):
    
    config = {
        "map": "SCS",
        "traffic_density": 0.1,
        "num_scenarios": 1,
        "start_seed": 12345,
        "manual_control": policy_type == "human",
        "use_render": True,
        "vehicle_config": {
            "show_lidar": True,
            "show_navi_mark": True,
            "show_line_to_navi_mark": True
        }
    }
    
    env = TopDownMetaDrive(config)
    
    try:
        obs, _ = env.reset()
        
        if policy_type == "noisy_expert":
            policy = NoisyExpertPolicy(env.agent)
            phase_name = "NOISY EXPERT DEMONSTRATION"
        elif policy_type == "irl":
            policy = NaiveIRLPolicy(env.agent)
            policy.iteration = (iteration - 1) // 2
            phase_name = f"NAIVE IRL MODEL (Learning Iteration {policy.iteration + 1})"
        else:
            policy = None
            phase_name = "HUMAN DEMONSTRATION"
        
        print(f"\n{'='*60}")
        print(f"{phase_name}")
        print(f"Iteration {iteration}/5")
        print(f"{'='*60}")
        
        if policy_type == "human":
            print("Use WASD keys to drive. Press ESC to end.")
        elif policy_type == "irl":
            print("Watch the naive AI try to drive (it starts knowing nothing!)")
        else:
            print("Watch the noisy expert drive poorly.")
        
        for step in range(500):
            
            if policy_type == "human":
                action = [0, 0]
            else:
                action = policy.act()
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            env.render(
                mode="top_down",
                text={
                    "Phase": phase_name,
                    "Iteration": f"{iteration}/5",
                    "Step": str(step + 1),
                    "Press ESC to continue": ""
                },
                film_size=(1200, 1200)
            )
            
            if terminated or truncated:
                break
                
        print(f"Demonstration completed after {step + 1} steps")
        
    except KeyboardInterrupt:
        print("Demonstration skipped by user")
    finally:
        env.close()


def main():
    print("Teaching Experiment")
    print("5 iterations: Noisy Expert -> Human -> Naive IRL -> Human -> Naive IRL")
    print("The IRL model starts knowing NOTHING about driving!")
    print("Map: SCS")
    print("\nPress Enter to start...")
    input()
    
    for i in range(1, 6):
        if i == 1:
            run_demonstration("noisy_expert", i)
        elif i % 2 == 0:
            run_demonstration("human", i)
        else:
            run_demonstration("irl", i)
        
        if i < 5:
            print(f"\nCompleted iteration {i}/5")
            response = input("Continue to next iteration? (y/n): ").lower().strip()
            if response != "y":
                break
    
    print("\nExperiment completed!")


if __name__ == "__main__":
    main()
