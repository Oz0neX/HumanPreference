#!/usr/bin/env python
"""
Debug script to test agent policies in isolation.
This helps identify if the issue is with the policy or the experiment framework.
"""

import numpy as np
from metadrive.envs.top_down_env import TopDownMetaDrive
from metadrive.obs.state_obs import LidarStateObservation
from naive_policies import get_naive_policy
from map_config import create_environment_config

def test_agent_policy(policy_type="simple_forward", steps=50):
    """Test a single agent policy to see if it works."""
    print(f"Testing {policy_type} policy...")
    
    # Create environment
    config, map_info = create_environment_config("curves", manual_control=False)
    env = TopDownMetaDrive(config)
    
    try:
        # Reset environment
        obs, _ = env.reset()
        print(f"Environment reset successful, obs type: {type(obs)}")
        
        # Create policy
        policy = get_naive_policy(policy_type, env.agent, random_seed=42)
        print(f"Created policy: {policy.__class__.__name__}")
        
        # Create state observation
        state_obs = LidarStateObservation(env.config.copy())
        print("State observation created")
        
        # Test a few steps
        for step in range(steps):
            try:
                # Get state and action
                state = state_obs.observe(env.agent)
                position = (env.agent.position[0], env.agent.position[1])
                action = policy.act()
                
                print(f"Step {step}: state shape {state.shape}, action {action}, position {position}")
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Render occasionally
                if step % 10 == 0:
                    env.render(
                        mode="top_down",
                        text={
                            "Debug Test": f"Step {step}",
                            "Policy": policy_type,
                            "Action": f"[{action[0]:.2f}, {action[1]:.2f}]",
                            "Reward": f"{reward:.3f}"
                        },
                        film_size=(800, 800)
                    )
                
                if terminated or truncated:
                    print(f"Episode ended at step {step} (terminated={terminated}, truncated={truncated})")
                    break
                    
            except Exception as e:
                print(f"Error at step {step}: {e}")
                break
                
        print(f"Test completed successfully for {policy_type}")
        return True
        
    except Exception as e:
        print(f"Test failed for {policy_type}: {e}")
        return False
    finally:
        env.close()

if __name__ == "__main__":
    print("Agent Policy Debug Test")
    print("=" * 40)
    
    # Test different policies
    policies = ["simple_forward", "random", "noisy_expert"]
    
    for policy_type in policies:
        print(f"\n--- Testing {policy_type} ---")
        success = test_agent_policy(policy_type, steps=30)
        print(f"Result: {'✓ PASS' if success else '✗ FAIL'}")
        print("-" * 40)
