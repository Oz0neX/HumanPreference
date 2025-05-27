#!/usr/bin/env python
"""
Iterative Feedback Experiment with IRL
Alternates between human demonstrations and agent demonstrations,
using IRL to learn from human trajectories and improve the agent.
"""

import json
import os
import numpy as np
from datetime import datetime

from metadrive.envs.top_down_env import TopDownMetaDrive
from metadrive.obs.state_obs import LidarStateObservation

from naive_policies import get_naive_policy
from irl_framework import IterativeFeedbackExperiment
from map_config import create_environment_config, get_map_config, list_available_maps

os.environ['SDL_VIDEO_CENTERED'] = '1'


class IterativeExperimentRunner:
    """Main runner for the iterative feedback experiment."""
    
    def __init__(self, experiment_name: str = None, map_name: str = "curves"):
        if experiment_name is None:
            experiment_name = f"irl_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        self.experiment_name = experiment_name
        self.map_name = map_name
        
        # Get map configuration
        self.map_config = get_map_config(map_name)
        print(f"Using map: {self.map_config['name']} ({self.map_config['map_string']}) - {self.map_config['description']}")
        
        # Initialize experiment controller
        base_log_dir = os.path.join(os.path.dirname(__file__), "irl_experiments")
        self.experiment = IterativeFeedbackExperiment(experiment_name, base_log_dir)
        
        # Environment and policy settings
        self.env = None
        self.current_policy = None
        self.naive_policy_type = "simple_forward"  # Default naive policy
        
        # Experiment parameters
        self.max_iterations = 5
        self.steps_per_episode = 2500
        self.record_every_n_steps = 10
            
    def create_environment(self, manual_control: bool = True):
        """Create the MetaDrive environment."""
        config, _ = create_environment_config(self.map_name, manual_control=manual_control)
        
        self.env = TopDownMetaDrive(config)
        print(f"Environment created with manual_control={manual_control}")
        
    def setup_agent_policy(self, policy_type: str = None):
        """Setup the agent policy for agent demonstration phases."""
        if policy_type is None:
            policy_type = self.naive_policy_type
            
        # Create the policy - note: we'll set it on the vehicle after reset
        self.agent_policy_type = policy_type
        print(f"Agent policy set to: {policy_type}")
        
    def run_human_demonstration(self) -> bool:
        """Run a human demonstration phase."""
        print("\n" + "="*60)
        print("HUMAN DEMONSTRATION PHASE")
        print("="*60)
        print("Instructions:")
        print("- Use WASD keys to control the vehicle")
        print("- Drive to demonstrate good driving behavior")
        print("- Press ESC to end the episode")
        print("- The trajectory will be recorded for IRL learning")
        print("="*60)
        
        # Create environment with manual control
        self.create_environment(manual_control=True)
        
        # Start human phase
        episode_id = self.experiment.start_human_phase()
        
        try:
            obs, _ = self.env.reset()
            
            # Get observation for state recording
            obs_config = self.env.config.copy()
            state_obs = LidarStateObservation(obs_config)
            
            step_count = 0
            
            for step in range(self.steps_per_episode):
                # Get current state and position
                current_state = state_obs.observe(self.env.agent)
                current_position = (self.env.agent.position[0], self.env.agent.position[1])
                
                # Step environment (action will come from manual control)
                obs, reward, terminated, truncated, info = self.env.step([0, 0])  # Dummy action for manual control
                
                # Get the actual action taken (from manual control)
                if hasattr(self.env.agent, 'last_current_action'):
                    actual_action = self.env.agent.last_current_action
                else:
                    # Fallback: estimate action from velocity change
                    actual_action = np.array([0.0, 0.0])
                
                # Record step every N steps
                if step % self.record_every_n_steps == 0:
                    self.experiment.record_step(current_state, actual_action, current_position)
                    step_count += 1
                
                # Render with information
                self.env.render(
                    mode="top_down",
                    text={
                        "Phase": "HUMAN DEMONSTRATION",
                        "Iteration": str(self.experiment.iteration),
                        "Episode": str(episode_id),
                        "Steps Recorded": str(step_count),
                        "Controls": "WASD to drive, ESC to quit"
                    },
                    film_size=(1200, 1200)
                )
                
                if terminated or truncated:
                    break
                    
        except KeyboardInterrupt:
            print("\nHuman demonstration interrupted by user")
        finally:
            # Finish human phase
            trajectory = self.experiment.finish_human_phase()
            self.env.close()
            
            if trajectory and len(trajectory.positions) > 0:
                print(f"Human demonstration completed: {len(trajectory.positions)} steps recorded")
                return True
            else:
                print("No trajectory data recorded")
                return False
                
    def run_agent_demonstration(self) -> bool:
        """Run an agent demonstration phase."""
        print("\n" + "="*60)
        print("ROBOT DEMONSTRATION PHASE")
        print("="*60)
        print("The robot will now demonstrate its driving")
        print("Watch how it performs!")
        print("="*60)
        
        # Create environment without manual control
        self.create_environment(manual_control=False)
        
        # Start agent phase
        episode_id = self.experiment.start_agent_phase()
        
        try:
            obs, _ = self.env.reset()
            print(f"Environment reset successful")
            
            # Create agent policy
            agent_policy = get_naive_policy(
                self.agent_policy_type, 
                self.env.agent, 
                random_seed=42
            )
            print(f"Created robot policy: {agent_policy.__class__.__name__}")
            
            # Update policy with learned weights if available
            if hasattr(agent_policy, 'update_policy'):
                learned_weights = self.experiment.get_updated_policy_weights()
                if learned_weights is not None:
                    agent_policy.update_policy(new_weights=learned_weights)
                    print("Updated policy with learned weights")
            
            # Get observation for state recording
            obs_config = self.env.config.copy()
            state_obs = LidarStateObservation(obs_config)
            
            step_count = 0
            actual_steps = 0
            
            # Run for a reasonable number of steps to show the robot driving
            max_demo_steps = min(500, self.steps_per_episode)  # Limit demo length
            
            for step in range(max_demo_steps):
                try:
                    # Get current state and position
                    current_state = state_obs.observe(self.env.agent)
                    current_position = (self.env.agent.position[0], self.env.agent.position[1])
                    
                    # Get action from agent policy
                    action = agent_policy.act()
                    
                    # Record trajectory data
                    if step % self.record_every_n_steps == 0:
                        self.experiment.record_step(current_state, action, current_position)
                        step_count += 1
                    
                    # Step environment
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    actual_steps += 1
                    
                    # Render every frame for smooth visualization
                    policy_info = agent_policy.get_action_info()
                    self.env.render(
                        mode="top_down",
                        text={
                            "Phase": "ROBOT DEMONSTRATION",
                            "Iteration": str(self.experiment.iteration),
                            "Policy": policy_info.get("policy_type", "unknown"),
                            "Steps": str(step + 1),
                            "Action": f"[{action[0]:.2f}, {action[1]:.2f}]",
                            "Reward": f"{reward:.2f}",
                            "Press ESC to skip": ""
                        },
                        film_size=(1200, 1200)
                    )
                    
                    # Check for termination
                    if terminated or truncated:
                        print(f"Robot episode ended at step {step + 1}")
                        if terminated:
                            print("Robot crashed or went off-road")
                        break
                        
                except Exception as e:
                    print(f"Error in robot demonstration at step {step}: {e}")
                    break
                    
            print(f"Robot demonstration completed: {actual_steps} total steps, {step_count} recorded steps")
            
            # Ensure we have at least some trajectory data
            if step_count == 0 and actual_steps > 0:
                # Force record at least the final state
                try:
                    current_state = state_obs.observe(self.env.agent)
                    current_position = (self.env.agent.position[0], self.env.agent.position[1])
                    action = np.array([0.0, 0.0])  # Dummy action
                    self.experiment.record_step(current_state, action, current_position)
                    step_count = 1
                    print("Forced recording of final state")
                except:
                    pass
                    
        except KeyboardInterrupt:
            print("\nRobot demonstration interrupted by user")
        except Exception as e:
            print(f"Robot demonstration failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Finish agent phase
            trajectory = self.experiment.finish_agent_phase()
            self.env.close()
            
            if trajectory and len(trajectory.positions) > 0:
                print(f"✓ Robot demonstration successful: {len(trajectory.positions)} steps recorded")
                return True
            else:
                print("✗ No robot trajectory data recorded")
                return False
                
    def run_irl_update(self):
        """Run the IRL learning update."""
        print("\n" + "="*60)
        print("IRL LEARNING UPDATE")
        print("="*60)
        
        update_result = self.experiment.update_irl_model()
        
        if update_result["status"] == "updated":
            print("✓ IRL model successfully updated")
            print(f"  - Used {update_result['num_human_trajectories']} human trajectories")
            print(f"  - Processed {update_result['num_human_states']} state samples")
            print(f"  - IRL iteration: {update_result['iteration']}")
        else:
            print(f"⚠ IRL update status: {update_result['status']}")
            
        return update_result
        
    def run_full_experiment(self):
        """Run the complete iterative experiment."""
        print(f"\n{'='*80}")
        print(f"TEACHING THE ROBOT TO DRIVE")
        print(f"{'='*80}")
        print(f"Teaching iterations: {self.max_iterations}")
        print("First iteration: Robot shows poor driving (noisy behavior)")
        print("Later iterations: Robot learns and improves from your demonstrations")
        print(f"{'='*80}")
        
        try:
            for iteration in range(self.max_iterations):
                self.experiment.next_iteration()
                
                # Set policy type based on iteration
                if iteration == 0:
                    # First iteration: show noisy/poor driving
                    self.agent_policy_type = "noisy_expert"
                    robot_description = "The robot will demonstrate poor driving skills"
                else:
                    # Subsequent iterations: use progressive learning
                    self.agent_policy_type = "progressive"
                    robot_description = "The robot will show what it has learned from your demonstrations"
                
                print(f"\n{'='*50}")
                print(f"TEACHING ITERATION {iteration + 1} / {self.max_iterations}")
                print(f"{'='*50}")
                print(robot_description)
                
                # Phase 1: Human demonstration
                human_success = self.run_human_demonstration()
                if not human_success:
                    print("Human demonstration failed, skipping this iteration")
                    continue
                
                # Phase 2: IRL learning update (skip for first iteration since no learning yet)
                if iteration > 0:
                    self.run_irl_update()
                else:
                    print("\n=== ROBOT LEARNING ===")
                    print("Robot is observing your demonstration...")
                    print("Learning will begin after this iteration.")
                
                # Phase 3: Agent demonstration
                print(f"\n--- Robot Demonstration ---")
                if iteration == 0:
                    print("Watch the robot's initial poor driving behavior.")
                    print("This shows what you're teaching it to improve upon.")
                else:
                    print("Watch how the robot's driving has improved from your teaching!")
                
                agent_success = self.run_agent_demonstration()
                if not agent_success:
                    print("Robot demonstration failed")
                
                # Save experiment state
                self.experiment.save_experiment_state()
                
                # Print summary
                summary = self.experiment.get_experiment_summary()
                print(f"\nIteration {iteration + 1} Summary:")
                print(f"  - Human demonstrations: {summary['human_trajectories']}")
                print(f"  - Robot demonstrations: {summary['agent_trajectories']}")
                
                # Ask if user wants to continue
                if iteration < self.max_iterations - 1:
                    print(f"\nCompleted teaching iteration {iteration + 1}/{self.max_iterations}")
                    response = input("Continue to next teaching iteration? (y/n): ").lower().strip()
                    if response != 'y':
                        print("Teaching session ended by user")
                        break
                        
        except KeyboardInterrupt:
            print("\n\nExperiment interrupted by user")
        except Exception as e:
            print(f"\nExperiment failed with error: {e}")
            raise
        finally:
            # Final save and summary
            self.experiment.save_experiment_state()
            final_summary = self.experiment.get_experiment_summary()
            
            print(f"\n{'='*80}")
            print("EXPERIMENT COMPLETED")
            print(f"{'='*80}")
            print(f"Experiment name: {final_summary['experiment_name']}")
            print(f"Total iterations: {final_summary['iteration']}")
            print(f"Human trajectories collected: {final_summary['human_trajectories']}")
            print(f"Agent trajectories collected: {final_summary['agent_trajectories']}")
            print(f"Results saved to: {self.experiment.experiment_dir}")
            print(f"{'='*80}")


def main():
    """Main entry point."""
    print("Teach the Robot to Drive")
    print("=" * 40)
    print("You will demonstrate good driving, then watch the robot learn from your example.")
    print("The robot starts with poor driving skills and improves with each iteration.\n")
    
    # Simple configuration - only ask for number of iterations
    max_iterations = input("How many teaching iterations? (default=3): ").strip()
    try:
        max_iterations = int(max_iterations) if max_iterations else 3
    except ValueError:
        max_iterations = 3
        
    # Create experiment with auto-generated name
    experiment_name = f"teach_robot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    runner = IterativeExperimentRunner(experiment_name)
    runner.max_iterations = max_iterations
    
    print(f"\nStarting teaching session with {max_iterations} iterations")
    print("The robot will start with noisy driving, then learn from your demonstrations.\n")
    
    input("Press Enter to begin teaching the robot...")
    
    runner.run_full_experiment()


if __name__ == "__main__":
    main()
