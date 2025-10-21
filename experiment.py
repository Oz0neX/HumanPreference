import tkinter as tk
from tkinter import Frame, Label, Button, messagebox
import numpy as np
import pandas as pd
from datetime import datetime
import os
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.examples.ppo_expert.numpy_expert import expert

from imitation.data import types
from imitation.data import serialize
import sys

IS_TEACHING_EXPERIMENT = True
NUM_TEACHING = 2
NUM_OPTIMAL = 1

# Discrete action configuration
DISCRETE_STEERING_DIM = 5
DISCRETE_THROTTLE_DIM = 5

class ContinuousReplayPolicy:
    """Continuous replay policy - exact copy from replay_record.py"""
    def __init__(self, trajectory):
        self.trajectory = trajectory
        self.step = 0

    def act(self):
        if self.step >= len(self.trajectory.acts):
            return [0.0, 0.0]
        action = self.trajectory.acts[self.step]
        self.step += 1
        return action

def get_discrete_action_values(steering_dim=DISCRETE_STEERING_DIM, throttle_dim=DISCRETE_THROTTLE_DIM):
    steering_unit = 2.0 / (steering_dim - 1)
    throttle_unit = 2.0 / (throttle_dim - 1)
    
    steering_values = [i * steering_unit - 1.0 for i in range(steering_dim)]
    throttle_values = [i * throttle_unit - 1.0 for i in range(throttle_dim)]
    
    return steering_values, throttle_values

def continuous_to_discrete_action(continuous_action, steering_values, throttle_values):
    steering, throttle = continuous_action[0], continuous_action[1]
    
    # Find closest discrete values
    steering_discrete = min(steering_values, key=lambda x: abs(x - steering))
    throttle_discrete = min(throttle_values, key=lambda x: abs(x - throttle))
    
    return np.array([steering_discrete, throttle_discrete], dtype=np.float32)

main_color = '#3e3e42'
secondary_color = '#252526'

# Check for --linux flag to enable embedded windows
USE_EMBEDDED_WINDOWS = "--linux" in sys.argv
print(f"Embedded windows mode: {USE_EMBEDDED_WINDOWS}")

class NoisyExpertPolicy:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.params = locals()

    def act(self):
        try:
            # For multi-discrete environment with DISCRETE_STEERING_DIM=5, DISCRETE_THROTTLE_DIM=5
            # Steering values [0,1,2,3,4] map to [-1.0, -0.5, 0.0, 0.5, 1.0]
            # Throttle values [0,1,2,3,4] map to [-1.0, -0.5, 0.0, 0.5, 1.0]

            # Simple noisy expert: slight left turn (steering index 3 = 0.5)
            # with full throttle (throttle index 4 = 1.0)
            # Add some noise to occasionally do different actions

            steering_noise = 0.3  # Probability of different steering
            throttle_noise = 0.2  # Probability of different throttle

            if np.random.random() < steering_noise:
                steering_idx = np.random.randint(0, 5)
            else:
                steering_idx = 3  # Prefer slight left turn

            if np.random.random() < throttle_noise:
                throttle_idx = np.random.randint(0, 5)
            else:
                throttle_idx = 4  # Prefer full throttle

            # Return as tuple of integers for multi-discrete action space
            return (steering_idx, throttle_idx)

        except Exception:
            # Fallback to random discrete actions
            return (np.random.randint(0, 5), np.random.randint(0, 5))

class RobotTeachingApp:
    def __init__(self, root):
        self.root = root
        self.root.configure(bg=main_color)
        self.center_window()

        self.current_iteration = 0
        self.teaching_counter = 0  # Separate counter for teaching trajectories (1, 2, 3...)
        self.part2_local_iter = 0  # Local iteration counter for Part 2
        self.part3_local_iter = 0  # Local iteration counter for Part 3
        self.env = None
        self.policy = None
        self.running = False
        self.current_part = 1
        self.part1_complete = False
        self.part2_complete = False  # Track completion of Part 2
        
        # Always put optimal first, regardless of IS_TEACHING_EXPERIMENT
        # Generate optimal phase sequence: human_demo for NUM_OPTIMAL times
        optimal_sequence = ["human_demo"] * NUM_OPTIMAL

        # Generate teaching phase sequence: agent-human alternating for NUM_TEACHING cycles
        teaching_sequence = ["noisy_expert"]
        for i in range(NUM_TEACHING):
            teaching_sequence.extend(["human"])

        # Combine: optimal first, then teaching (same order regardless of experiment type)
        self.phase_sequence = optimal_sequence + teaching_sequence
        self.total_iterations = len(self.phase_sequence)

        # Part 1 is optimal demonstrations (first iterations)
        self.part1_iterations = len(optimal_sequence)
        # Part 2 is teaching demonstrations (next set of iterations)
        self.part2_iterations = len(teaching_sequence)
        # Part 3 is additional teaching demonstrations on opposite map (final set)
        self.part3_iterations = len(teaching_sequence)

        # Separate phase sequences for Parts 2 and 3 (same as teaching_sequence)
        self.part2_phase_sequence = teaching_sequence
        self.part3_phase_sequence = teaching_sequence

        if IS_TEACHING_EXPERIMENT:
            self.teaching_seed = 12345  # Different seed for teaching vs demo
            self.demo_seed = 12345
        else:
            self.demo_seed = 54321
            self.teaching_seed = 54321

        # Trajectory recording variables for gym format
        self.current_episode_obs = []
        self.current_episode_acts = []
        self.current_episode_infos = []
        self.recording_started = False
        self.trajectory_saved = False
        self.start_position = None
        self.trajectory_data_dir = "trajectory_data"
        
        # Create trajectory data directory
        os.makedirs(self.trajectory_data_dir, exist_ok=True)

        self.setup_ui()

    def center_window(self):
        self.root.update_idletasks()
        screen_w, screen_h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        w, h = 1200, 800
        x, y = (screen_w // 2) - (w // 2), (screen_h // 2) - (h // 2)
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def setup_ui(self):
        main_frame = Frame(self.root, bg=main_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        def create_label(parent, text, font_size, bold=False, side=None, fill=None):
            label = Label(parent, text=text, font=("Arial", font_size, "bold" if bold else "normal"),
                          bg=main_color, fg="white", justify=tk.CENTER, wraplength=700)
            if side: label.pack(side=side, fill=fill)
            else: label.pack(pady=(0, 15))
            return label

        status_frame = Frame(main_frame, bg=main_color)
        status_frame.pack(fill=tk.X, pady=(0, 15))

        self.sim_frame = Frame(main_frame, bg=secondary_color, relief=tk.SUNKEN, borderwidth=3)
        self.sim_frame.pack(pady=(0, 10), padx=20, fill=tk.BOTH, expand=True)
        self.sim_frame.pack_propagate(False)
        
        part_name = "wait for further instructions, when ready," if IS_TEACHING_EXPERIMENT else "wait for further instructions, when ready,"
        self.sim_placeholder = create_label(self.sim_frame, "Please "+part_name+"\n\nClick 'Start Experiment' to begin Part 1", 14)
        self.sim_placeholder.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        self.status_label = Label(main_frame, text="...", bg=main_color, fg="white", font=('Arial', 12))
        self.status_label.pack(pady=(5, 0))

        button_frame = Frame(main_frame, bg=main_color)
        button_frame.pack(pady=15)
        self.buttons = {}
        btn_configs = {
            "Start Experiment": {"bg": "#4CAF50", "cmd": self.start_experiment},
            "Begin Part 2": {"bg": "#FF9800", "cmd": self.continue_to_part2, "state": tk.DISABLED},
            "Begin Part 3": {"bg": "#9C27B0", "cmd": self.continue_to_part3, "state": tk.DISABLED},
            "Stop": {"bg": "#F44336", "cmd": self.stop_simulation, "state": tk.DISABLED}
        }
        for name, config in btn_configs.items():
            btn = Button(button_frame, text=name, font=("Arial", 14, "bold"),
                         fg="white", bg=config["bg"], padx=25, pady=12, relief=tk.RAISED,
                         borderwidth=3, command=config["cmd"], state=config.get("state", tk.NORMAL))
            btn.pack(side=tk.LEFT, padx=8)
            self.buttons[name.lower()] = btn

    def update_ui_state(self, is_running, part1_complete=False, part2_complete=False):
        self.buttons["start experiment"].config(state=tk.DISABLED if is_running else tk.NORMAL, bg="#CCCCCC" if is_running else "#4CAF50")
        self.buttons["begin part 2"].config(state=tk.NORMAL if part1_complete else tk.DISABLED, bg="#FF9800" if part1_complete else "#CCCCCC")
        self.buttons["begin part 3"].config(state=tk.NORMAL if part2_complete else tk.DISABLED, bg="#9C27B0" if part2_complete else "#CCCCCC")
        self.buttons["stop"].config(state=tk.NORMAL if is_running else tk.DISABLED, bg="#F44336" if is_running else "#CCCCCC")
        if is_running:
            self.sim_placeholder.place_forget()
        else:
            self.sim_placeholder.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def start_experiment(self):
        self.running = True
        self.current_iteration = 1
        self.update_ui_state(is_running=True)
        self.status_label.config(text="...")
        policy_type = self.phase_sequence[0]
        self.run_demonstration(policy_type)

    def next_phase(self):
        # Save trajectory if we have recorded data and haven't saved it yet
        if self.recording_started and len(self.current_episode_obs) > 0 and not self.trajectory_saved:
            self.save_trajectory()
            self.trajectory_saved = True

        if self.current_part == 1 and self.current_iteration >= self.part1_iterations:
            self.complete_part1()
            return

        if self.current_part == 2 and self.part2_local_iter >= self.part2_iterations:
            self.complete_part2()
            return

        if self.current_part == 3 and self.part3_local_iter >= self.part3_iterations:
            self.stop_simulation(completed=True)
            return

        # Increment local iteration and get next policy type for current part
        if self.current_part == 2:
            self.part2_local_iter += 1
            policy_type = self.part2_phase_sequence[self.part2_local_iter - 1]
        elif self.current_part == 3:
            self.part3_local_iter += 1
            policy_type = self.part3_phase_sequence[self.part3_local_iter - 1]
        else:
            # Part 1 uses global counter
            self.current_iteration += 1
            policy_type = self.phase_sequence[self.current_iteration - 1]

        self.run_demonstration(policy_type)

    def complete_part1(self):
        self.running = False
        if self.env:
            self.env.close()
            self.env = None
        self.part1_complete = True
        self.update_ui_state(is_running=False, part1_complete=True)
        part_name = "wait for further instructions, when ready," if not IS_TEACHING_EXPERIMENT else "wait for further instructions, when ready,"
        self.sim_placeholder.config(text="Now please "+part_name+"\n\n Click 'Begin Part 2'", font=("Arial", 18, "bold"))

    def complete_part2(self):
        self.running = False
        if self.env:
            self.env.close()
            self.env = None
        self.part2_complete = True
        self.update_ui_state(is_running=False, part1_complete=True, part2_complete=True)
        part_name = "wait for further instructions, when ready," if not IS_TEACHING_EXPERIMENT else "wait for further instructions, when ready,"
        self.sim_placeholder.config(text="Now please "+part_name+"\n\n Click 'Begin Part 3'", font=("Arial", 18, "bold"))

    def continue_to_part2(self):
        self.current_part = 2
        self.part2_local_iter = 1
        self.teaching_counter = 0  # Reset teaching counter for part 2 (teaching trajectories start from 1)
        self.running = True
        self.part1_complete = False
        self.update_ui_state(is_running=True)

        policy_type = self.part2_phase_sequence[self.part2_local_iter - 1]
        self.run_demonstration(policy_type)

    def continue_to_part3(self):
        self.current_part = 3
        self.part3_local_iter = 1
        self.teaching_counter = 0  # Reset teaching counter for part 3 (teaching trajectories start from 1)
        self.running = True
        self.part2_complete = False
        self.update_ui_state(is_running=True)

        policy_type = self.part3_phase_sequence[self.part3_local_iter - 1]
        self.run_demonstration(policy_type)

    def stop_simulation(self, completed=False):
        # Save any remaining trajectory data
        if self.recording_started and len(self.current_episode_obs) > 0:
            self.save_trajectory()
            
        self.running = False
        if self.env:
            self.env.close()
            self.env = None
        self.update_ui_state(is_running=False)
        if not completed:
            part_name = "wait for further instructions, when ready," if IS_TEACHING_EXPERIMENT else "wait for further instructions, when ready,"
            self.sim_placeholder.config(text="Please "+part_name+"\n\nClick 'Start Experiment' to begin")

    def get_key_states(self):
        """Capture current key states using MetaDrive's native InputState"""
        try:
            controller = self.env.engine.get_policy(self.env.agent.id).controller
            inputs = controller.inputs
            
            return {
                'forward': int(inputs.isSet('forward')),
                'left': int(inputs.isSet('turnLeft')),
                'right': int(inputs.isSet('turnRight')),
                'brake': int(inputs.isSet('reverse'))
            }
        except:
            return {'forward': 0, 'left': 0, 'right': 0, 'brake': 0}

    def run_demonstration(self, policy_type):
        # Reset trajectory recording
        self.current_episode_obs = []
        self.current_episode_acts = []
        self.current_episode_infos = []
        self.recording_started = False
        self.trajectory_saved = False
        self.start_position = None
        self.current_policy_type = policy_type
        
        # Determine if this is an optimal demonstration (human_demo)
        is_optimal = policy_type == "human_demo"
        
        # For optimal demonstrations, try to reuse environment with reset
        # For teaching demonstrations, always close and recreate
        if is_optimal and self.env is not None:
            # Reset environment for optimal demonstrations
            if IS_TEACHING_EXPERIMENT:
                seed = self.demo_seed  # Optimal demos use demo_seed
            else:
                seed = self.demo_seed  # Optimal demos use demo_seed
            
            obs = self.env.reset(seed=seed)[0]
            self.env.manual_control = True  # Optimal demos are always manual
        else:
            # Close and recreate environment for teaching demonstrations or first optimal demo
            if self.env: 
                self.env.close()
                self.env = None
            
            if IS_TEACHING_EXPERIMENT:
                if self.current_part == 3:
                    seed = 54321  # Opposite seed for Part 3
                else:
                    seed = self.teaching_seed if self.current_iteration <= self.part1_iterations else self.demo_seed
            else:
                if self.current_part == 3:
                    seed = 12345  # Opposite seed for Part 3 (when not teaching experiment)
                else:
                    seed = self.demo_seed if self.current_iteration <= self.part1_iterations else self.teaching_seed
            
            is_manual = policy_type in ["human", "human_demo"]

            if USE_EMBEDDED_WINDOWS:
                # Use embedded windows for Linux compatibility
                # Get the size of the simulation frame for embedding
                self.root.update_idletasks()
                frame_width = self.sim_frame.winfo_width() - 6  # Account for border
                frame_height = self.sim_frame.winfo_height() - 6

                config = {
                    "map": "SCS",
                    "traffic_density": 0.1,
                    "num_scenarios": 1,
                    "start_seed": seed,
                    "manual_control": is_manual,
                    "use_render": True,
                    "window_size": (frame_width, frame_height),
                    "parent_window": self.sim_frame.winfo_id(),  # Embed in the sim_frame
                    "vehicle_config": {"show_navi_mark": True, "show_line_to_navi_mark": True},
                    # Termination conditions - disable collision-based endings
                    "on_continuous_line_done": False,
                    "out_of_route_done": False,
                    "crash_vehicle_done": False,
                    "crash_object_done": False,
                    "discrete_action": True,
                    "discrete_steering_dim": DISCRETE_STEERING_DIM,
                    "discrete_throttle_dim": DISCRETE_THROTTLE_DIM,
                    "use_multi_discrete": True
                }
            else:
                # Use separate windows (default Windows behavior)
                config = {
                    "map": "SCS",
                    "traffic_density": 0.1,
                    "num_scenarios": 1,
                    "start_seed": seed,
                    "manual_control": is_manual,
                    "use_render": True,
                    "window_size": (800, 600),
                    "multi_thread_render": False,
                    "vehicle_config": {"show_navi_mark": False, "show_line_to_navi_mark": True},
                    # Termination conditions - disable collision-based endings to allow continued driving
                    "on_continuous_line_done": False,
                    "out_of_route_done": False,
                    "crash_vehicle_done": False,
                    "crash_object_done": False,
                    "discrete_action": True,
                    "discrete_steering_dim": DISCRETE_STEERING_DIM,
                    "discrete_throttle_dim": DISCRETE_THROTTLE_DIM,
                    "use_multi_discrete": True
                }

            self.env = MetaDriveEnv(config)
            obs = self.env.reset()[0]

        if policy_type == "noisy_expert":
            # Use EXACT continuous replay from replay_record.py - separate environment
            trajectory_path = self.get_trajectory_path_for_iteration()
            if trajectory_path:
                print(f"Starting continuous trajectory replay...")
                # Close existing environment first to avoid engine conflicts
                if self.env:
                    self.env.close()
                    self.env = None
                # Perform continuous replay using EXACT code from replay_record.py
                self.perform_continuous_replay(trajectory_path)
                # After continuous replay, move to next phase
                self.root.after(2000, self.next_phase)  # Give time for auto-close message
                return
            else:
                print("No trajectory found, falling back to NoisyExpertPolicy")
                self.policy = NoisyExpertPolicy(self.env.agent)
        else:
            self.policy = None
        
        self.simulation_loop()

    def simulation_loop(self):
        if not self.running: 
            return
        try:
            if self.current_policy_type in ["human", "human_demo"]:
                obs, reward, terminated, truncated, info = self.env.step([0, 0])
                
                steering = info.get('steering', 0.0) if info else 0.0
                throttle = info.get('acceleration', 0.0) if info else 0.0
                action = [steering, throttle]
            else:
                action = self.policy.act()
                obs, reward, terminated, truncated, info = self.env.step(action)
            
            if self.current_policy_type in ["human", "human_demo"]:
                # Start recording once the vehicle moves
                if not self.recording_started:
                    current_position = np.array([self.env.agent.position[0], self.env.agent.position[1]])
                    if self.start_position is None:
                        self.start_position = current_position.copy()
                    elif np.linalg.norm(current_position - self.start_position) >= 0.1:
                        self.recording_started = True
                        print("Started recording trajectory.")

                # If recording, capture data in gym format
                if self.recording_started:
                    current_obs = obs.copy()

                    steering = info.get('steering', 0.0) if info else 0.0
                    throttle = info.get('acceleration', 0.0) if info else 0.0
                    
                    # Convert to discrete action values in range [-1, 1]
                    steering_values, throttle_values = get_discrete_action_values()
                    continuous_action = np.array([steering, throttle], dtype=np.float32)
                    discrete_action = continuous_to_discrete_action(continuous_action, steering_values, throttle_values)
                    
                    self.current_episode_obs.append(current_obs)
                    self.current_episode_acts.append(discrete_action)
                    self.current_episode_infos.append({})
            
            # Check for trajectory replay completion
            if hasattr(self.policy, 'is_trajectory_complete') and self.policy.is_trajectory_complete():
                print("Trajectory replay complete, auto-closing experiment...")
                self.root.after(1000, lambda: self.root.destroy())  # Close after 1 second delay
                return

            if terminated or truncated:
                self.root.after(500, self.next_phase)
            else:
                self.root.after(20, self.simulation_loop)
        except Exception as e:
            print(f"Error in simulation loop: {e}")
            self.stop_simulation()
    
    def save_trajectory(self):
        """Save the current trajectory using imitation library format"""
        if len(self.current_episode_obs) == 0:
            return
            
        try:
            # Remove the final action to match imitation library expectations
            # (one more observation than actions)
            if len(self.current_episode_acts) > 0:
                actions = np.array(self.current_episode_acts[:-1])  # Remove last action
                infos = np.array(self.current_episode_infos[:-1])   # Remove last info to match actions
            else:
                actions = np.array([])
                infos = np.array([])
                
            # Create Trajectory object
            trajectory = types.Trajectory(
                obs=np.array(self.current_episode_obs),
                acts=actions,
                infos=infos,
                terminal=True
            )
            
            # Determine save path based on policy type and current seed used
            if self.current_policy_type == "human_demo":
                demo_type = "optimal"
            else:
                demo_type = "teaching"
            
            # Get the actual seed that was used for this environment
            current_seed = self.env.current_seed if self.env else None
            if current_seed is None:
                # Fallback: determine seed based on current phase
                if IS_TEACHING_EXPERIMENT:
                    current_seed = self.teaching_seed if self.current_iteration <= self.part1_iterations else self.demo_seed
                else:
                    current_seed = self.demo_seed if self.current_iteration <= self.part1_iterations else self.teaching_seed
            
            # Create directory structure: trajectory_data/map_XXXXX/demo_type/
            save_dir = os.path.join(self.trajectory_data_dir, f"map_{current_seed}", demo_type)
            os.makedirs(save_dir, exist_ok=True)
            
            # Create timestamp for unique filename
            timestamp = datetime.now().strftime("%m%d_%H%M%S_%f")[:-3]  # Include milliseconds to avoid duplicates
            save_path = os.path.join(save_dir, f"trajectory_{timestamp}")
            
            # Save trajectory using imitation library
            serialize.save(save_path, [trajectory])
            
            print(f"Saved {demo_type} trajectory with {len(self.current_episode_obs)} steps to {save_path} (seed: {current_seed})")
            print(f"Actions are discrete values in range [-1, 1]: {actions.shape if len(actions) > 0 else 'No actions'}")
            
        except Exception as e:
            print(f"Error saving trajectory: {e}")

    def perform_continuous_replay(self, trajectory_path):
        """Perform continuous replay using EXACT code from replay_record.py"""
        print(f"Performing continuous replay of: {trajectory_path}")

        # Use EXACT config from replay_record.py (continuous replay mode)
        config = {
            "map": "SCS",
            "traffic_density": 0.1,
            "start_seed": 12345,  # Use fixed seed for replay consistency
            "manual_control": False,  # False for replay
            "use_render": True,
            "vehicle_config": {"show_navi_mark": False},
            "discrete_action": False,      # CONTINUOUS actions
            "use_multi_discrete": False,   # CONTINUOUS actions
            "on_continuous_line_done": False,
            "out_of_route_done": False,
            "crash_vehicle_done": False,
            "crash_object_done": False,
        }

        # Load the saved trajectory - EXACT same code from replay_record.py
        trajectories = serialize.load(trajectory_path)
        trajectory = trajectories[0]
        print(f"Loaded trajectory with {len(trajectory.obs)} observations and {len(trajectory.acts)} actions")

        print("Starting continuous replay phase...")

        # Create continuous environment for replay
        env = MetaDriveEnv(config)
        obs, info = env.reset()

        # Use EXACT ContinuousReplayPolicy from replay_record.py
        policy = ContinuousReplayPolicy(trajectory)

        print("Replaying trajectory... Close window when done or wait for auto-close.")

        while True:
            action = policy.act()
            obs, reward, terminated, truncated, info = env.step(action)

            # Check if trajectory is complete
            if policy.step >= len(trajectory.acts):
                print("Continuous trajectory replay complete, auto-closing...")
                env.close()
                return True  # Signal completion

            if terminated or truncated:
                break

        env.close()
        print("Continuous replay ended due to termination")
        return False

    def get_trajectory_path_for_iteration(self):
        """Get the trajectory path for current iteration using the replay_record.py saved trajectories"""
        # Increment teaching counter for each teaching trajectory request
        self.teaching_counter += 1

        # Look in the 'recorded' directory where replay_record.py saves trajectories
        recorded_dir = "recorded"

        if not os.path.exists(recorded_dir):
            print(f"Recorded directory not found: {recorded_dir}")
            return None

        # First, try to find trajectory with expected naming pattern: {counter}_{seed}_trajectory
        if IS_TEACHING_EXPERIMENT:
            current_seed = self.teaching_seed if self.current_part == 2 else 54321  # Use teaching seed for Part 2, opposite seed for Part 3
        else:
            current_seed = 54321  # base seed for opposite calculation
        expected_pattern = f"{self.teaching_counter}_{current_seed}_trajectory"

        expected_path = os.path.join(recorded_dir, expected_pattern)
        if os.path.exists(expected_path):
            print(f"Found expected trajectory: {expected_path}")
            return expected_path

        # Fallback: find the most recent trajectory if expected pattern not found
        all_items = []
        for item in os.listdir(recorded_dir):
            item_path = os.path.join(recorded_dir, item)
            if os.path.isdir(item_path):
                all_items.append(item)

        if not all_items:
            print(f"No trajectory directories found in: {recorded_dir}")
            return None

        # Sort by modification time (most recent first)
        all_items.sort(key=lambda x: os.path.getmtime(os.path.join(recorded_dir, x)), reverse=True)
        selected_dir = all_items[0]

        trajectory_path = os.path.join(recorded_dir, selected_dir)
        print(f"Expected trajectory {expected_pattern} not found, using most recent: {trajectory_path}")

        return trajectory_path


if __name__ == "__main__":
    root = tk.Tk()
    app = RobotTeachingApp(root)
    root.mainloop()
