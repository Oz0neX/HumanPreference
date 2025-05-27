#!/usr/bin/env python
"""
Test script for naive policies.
This script allows you to test different naive policies to see how they behave.
"""

import json
import os
import numpy as np
from metadrive.envs.top_down_env import TopDownMetaDrive
from naive_policies import get_naive_policy
from map_config import create_environment_config, get_map_config

os.environ['SDL_VIDEO_CENTERED'] = '1'


def test_policy(policy_type: str, steps: int = 500, map_name: str = "curves"):
    """Test a specific naive policy."""
    print(f"\n{'='*60}")
    print(f"TESTING POLICY: {policy_type.upper()}")
    print(f"{'='*60}")
    
    # Use the new map configuration system
    config, map_info = create_environment_config(map_name, manual_control=False)
    
    print(f"Using map: {map_info['name']} ({map_info['map_string']}) - {map_info['description']}")
    
    env = TopDownMetaDrive(config)
    
    try:
        obs, _ = env.reset()
        
        # Create policy
        policy = get_naive_policy(policy_type, env.agent, random_seed=42)
        print(f"Created {policy_type} policy: {policy.__class__.__name__}")
        
        step_count = 0
        total_reward = 0
        
        for step in range(steps):
            # Get action from policy
            action = policy.act()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # Get policy info for display
            policy_info = policy.get_action_info()
            
            # Render with information
            env.render(
                mode="top_down",
                text={
                    "Policy Type": policy_type,
                    "Policy Class": policy.__class__.__name__,
                    "Step": str(step_count),
                    "Action": f"[{action[0]:.2f}, {action[1]:.2f}]",
                    "Reward": f"{reward:.3f}",
                    "Total Reward": f"{total_reward:.2f}",
                    "Policy Info": str(policy_info.get("policy_type", "unknown")),
                    "Learning Iter": str(policy_info.get("learning_iteration", 0)),
                    "Press ESC to stop": ""
                },
                film_size=(1200, 1200)
            )
            
            if terminated or truncated:
                print(f"Episode ended at step {step_count}")
                break
                
        print(f"Test completed: {step_count} steps, total reward: {total_reward:.2f}")
        return True
        
    except KeyboardInterrupt:
        print(f"\nTest interrupted by user at step {step_count}")
        return True
    except Exception as e:
        print(f"Test failed with error: {e}")
        return False
    finally:
        env.close()


def main():
    """Main test function."""
    print("Naive Policy Test Script")
    print("This script lets you test different naive policies to see how they behave.")
    
    available_policies = ["random", "noisy_expert", "simple_forward", "progressive"]
    
    while True:
        print(f"\nAvailable policies:")
        for i, policy in enumerate(available_policies, 1):
            print(f"{i}. {policy}")
        print("0. Exit")
        
        choice = input("\nChoose a policy to test (0-4): ").strip()
        
        if choice == "0":
            print("Exiting...")
            break
        elif choice in ["1", "2", "3", "4"]:
            policy_index = int(choice) - 1
            policy_type = available_policies[policy_index]
            
            steps = input(f"Number of steps to test (default=500): ").strip()
            try:
                steps = int(steps) if steps else 500
            except ValueError:
                steps = 500
                
            print(f"\nTesting {policy_type} policy for {steps} steps...")
            success = test_policy(policy_type, steps)
            
            if success:
                print(f"✓ {policy_type} policy test completed")
            else:
                print(f"✗ {policy_type} policy test failed")
        else:
            print("Invalid choice. Please enter 0-4.")


if __name__ == "__main__":
    main()
