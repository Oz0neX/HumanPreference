#!/usr/bin/env python
"""
Simple agent demonstration to test if the basic setup works.
"""

import numpy as np
from metadrive.envs.top_down_env import TopDownMetaDrive
from naive_policies import get_naive_policy
from map_config import create_environment_config

def simple_agent_test():
    """Test basic agent functionality."""
    print("Testing basic agent demonstration...")
    
    # Create environment
    config, map_info = create_environment_config("curves", manual_control=False)
    env = TopDownMetaDrive(config)
    
    try:
        # Reset environment
        obs, _ = env.reset()
        print(f"Environment reset successful")
        
        # Create noisy expert policy
        policy = get_naive_policy("noisy_expert", env.agent, random_seed=42)
        print(f"Created policy: {policy.__class__.__name__}")
        
        # Run for a limited number of steps
        for step in range(100):
            # Get action
            action = policy.act()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render every 10 steps
            if step % 10 == 0:
                env.render(
                    mode="top_down",
                    text={
                        "Test": "Simple Agent Demo",
                        "Step": str(step),
                        "Action": f"[{action[0]:.2f}, {action[1]:.2f}]",
                        "Policy": "noisy_expert"
                    },
                    film_size=(800, 800)
                )
                print(f"Step {step}: action={action}, terminated={terminated}, truncated={truncated}")
            
            if terminated or truncated:
                print(f"Episode ended at step {step}")
                break
                
        print("Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        env.close()

if __name__ == "__main__":
    simple_agent_test()
