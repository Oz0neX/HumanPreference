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


IS_TEACHING_EXPERIMENT = False
NUM_TEACHING = 15
NUM_OPTIMAL = 0

main_color = '#3e3e42'
secondary_color = '#252526'

class NoisyExpertPolicy:
    def __init__(self, vehicle, steering_noise=0.3, throttle_noise=0.2, corruption_prob=0.15):
        self.vehicle = vehicle
        self.params = locals()

    def act(self):
        if np.random.random() < self.params["corruption_prob"]:
            return np.random.uniform(-1, 1, size=2)
        try:
            # Instead of using expert, just return a simple turn action
            # This will make the agent turn left with some noise
            base_action = np.array([0.5, 1.0])  # Left turn, full throttle
            noise = np.random.normal(0, [self.params["steering_noise"], self.params["throttle_noise"]])
            return np.clip(base_action, -1, 1)
        except Exception:
            return np.random.uniform(-1, 1, size=2)



class RobotTeachingApp:
    def __init__(self, root):
        self.root = root
        self.root.configure(bg=main_color)
        self.center_window()

        self.current_iteration = 0
        self.env = None
        self.policy = None
        self.running = False
        self.current_part = 1
        self.part1_complete = False
        
        if IS_TEACHING_EXPERIMENT:
            # Generate teaching phase sequence: agent-human alternating for NUM_TEACHING cycles
            teaching_sequence = []
            for i in range(NUM_TEACHING):
                teaching_sequence.extend(["noisy_expert", "human"])
            
            # Generate optimal phase sequence: human_demo for NUM_OPTIMAL times
            optimal_sequence = ["human_demo"] * NUM_OPTIMAL
            
            # Combine: teaching first, then optimal
            self.phase_sequence = teaching_sequence + optimal_sequence
            self.total_iterations = len(self.phase_sequence)
            self.part1_iterations = len(teaching_sequence)
            self.part2_iterations = len(optimal_sequence)
            self.teaching_seed = 12345
            self.demo_seed = 54321
        else:
            # Generate optimal phase sequence: human_demo for NUM_OPTIMAL times
            optimal_sequence = ["human_demo"] * NUM_OPTIMAL
            
            # Generate teaching phase sequence: agent-human alternating for NUM_TEACHING cycles
            teaching_sequence = []
            for i in range(NUM_TEACHING):
                teaching_sequence.extend(["noisy_expert", "human"])
            
            # Combine: optimal first, then teaching
            self.phase_sequence = optimal_sequence + teaching_sequence
            self.total_iterations = len(self.phase_sequence)
            self.part1_iterations = len(optimal_sequence)
            self.part2_iterations = len(teaching_sequence)
            self.demo_seed = 12345
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
        
        part_name = "TEACH the robot how to drive" if IS_TEACHING_EXPERIMENT else "perform a normal drive"
        self.sim_placeholder = create_label(self.sim_frame, "Please "+part_name+".\n\nClick 'Start Experiment' to begin Part 1", 14)
        self.sim_placeholder.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        self.status_label = Label(main_frame, text="Ready to begin experiment", bg=main_color, fg="white", font=('Arial', 12))
        self.status_label.pack(pady=(5, 0))

        button_frame = Frame(main_frame, bg=main_color)
        button_frame.pack(pady=15)
        self.buttons = {}
        btn_configs = {
            "Start": {"bg": "#4CAF50", "cmd": self.start_experiment},
            "Next": {"bg": "#2196F3", "cmd": self.next_phase, "state": tk.DISABLED},
            "Continue": {"bg": "#FF9800", "cmd": self.continue_to_part2, "state": tk.DISABLED},
            "Stop": {"bg": "#F44336", "cmd": self.stop_simulation, "state": tk.DISABLED}
        }
        for name, config in btn_configs.items():
            text = f"{name} Phase" if name in ["Start", "Next"] else ("Begin Part 2" if name == "Continue" else name)
            btn = Button(button_frame, text=text, font=("Arial", 14, "bold"),
                         fg="white", bg=config["bg"], padx=25, pady=12, relief=tk.RAISED,
                         borderwidth=3, command=config["cmd"], state=config.get("state", tk.NORMAL))
            btn.pack(side=tk.LEFT, padx=8)
            self.buttons[name.lower()] = btn

    def update_ui_state(self, is_running, part1_complete=False):
        self.buttons["start"].config(state=tk.DISABLED if is_running else tk.NORMAL, bg="#CCCCCC" if is_running else "#4CAF50")
        self.buttons["next"].config(state=tk.NORMAL if is_running else tk.DISABLED, bg="#2196F3" if is_running else "#CCCCCC")
        self.buttons["continue"].config(state=tk.NORMAL if part1_complete else tk.DISABLED, bg="#FF9800" if part1_complete else "#CCCCCC")
        self.buttons["stop"].config(state=tk.NORMAL if is_running else tk.DISABLED, bg="#F44336" if is_running else "#CCCCCC")
        if is_running: 
            self.sim_placeholder.place_forget()
        else: 
            self.sim_placeholder.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def start_experiment(self):
        self.running = True
        self.current_iteration = 1
        self.update_ui_state(is_running=True)
        self.status_label.config(text="Starting experiment...")
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
        
        if self.current_part == 2 and (self.current_iteration - self.part1_iterations) >= self.part2_iterations:
            self.stop_simulation(completed=True)
            return
        
        self.current_iteration += 1
        policy_type = self.phase_sequence[self.current_iteration - 1]
        self.run_demonstration(policy_type)

    def complete_part1(self):
        self.running = False
        if self.env:
            try:
                self.env.close()
            except Exception as e:
                print(f"Warning: Error closing environment: {e}")
            finally:
                self.env = None
        self.part1_complete = True
        self.update_ui_state(is_running=False, part1_complete=True)
        part_name = "TEACH the robot how to drive" if not IS_TEACHING_EXPERIMENT else "perform a normal drive"
        self.sim_placeholder.config(text="Now please "+part_name+".\n\n Click 'Begin Part 2'", font=("Arial", 18, "bold"))

    def continue_to_part2(self):
        self.current_part = 2
        self.current_iteration = self.part1_iterations + 1
        self.running = True
        self.part1_complete = False
        self.update_ui_state(is_running=True)
        
        policy_type = self.phase_sequence[self.current_iteration - 1]
        self.run_demonstration(policy_type)

    def stop_simulation(self, completed=False):
        # Save any remaining trajectory data
        if self.recording_started and len(self.current_episode_obs) > 0:
            self.save_trajectory()
            
        self.running = False
        if self.env:
            try:
                self.env.close()
            except Exception as e:
                print(f"Warning: Error closing environment: {e}")
            finally:
                self.env = None
        self.update_ui_state(is_running=False)
        if not completed:
            part_name = "perform a normal drive" if IS_TEACHING_EXPERIMENT else "teach the robot how to drive"
            self.sim_placeholder.config(text="Please "+part_name+".\n\nClick 'Start Experiment' to begin")

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
                seed = self.teaching_seed if self.current_iteration <= self.part1_iterations else self.demo_seed
            else:
                seed = self.demo_seed if self.current_iteration <= self.part1_iterations else self.teaching_seed
            
            is_manual = policy_type in ["human", "human_demo"]
            
            config = {
                "map": "SCS", 
                "traffic_density": 0.1, 
                "num_scenarios": 1, 
                "start_seed": seed,
                "manual_control": is_manual, 
                "use_render": True, 
                "window_size": (800, 600),
                "multi_thread_render": False,
                "vehicle_config": {"show_navi_mark": True, "show_line_to_navi_mark": True},
                "on_continuous_line_done": False,
                "out_of_route_done": False,
                "crash_vehicle_done": False,
                "crash_object_done": False,
                "discrete_action": True
            }
            
            self.env = MetaDriveEnv(config)
            obs = self.env.reset()[0]

        if policy_type == "noisy_expert": 
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
                    current_act = np.array([steering, throttle], dtype=np.float32)
                    
                    self.current_episode_obs.append(current_obs)
                    self.current_episode_acts.append(current_act)
                    self.current_episode_infos.append({})
            
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
            
        except Exception as e:
            print(f"Error saving trajectory: {e}")

if __name__ == "__main__": 
    root = tk.Tk()
    app = RobotTeachingApp(root)
    root.mainloop()
