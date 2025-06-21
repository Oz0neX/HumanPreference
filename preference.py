import tkinter as tk
from tkinter import Frame, Label, Button, messagebox
import numpy as np
import pandas as pd
from datetime import datetime
import os
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.examples.ppo_expert.numpy_expert import expert

IS_TEACHING_EXPERIMENT = True

class NoisyExpertPolicy:
    def __init__(self, vehicle, steering_noise=0.3, throttle_noise=0.2, corruption_prob=0.15):
        self.vehicle = vehicle
        self.params = locals()

    def act(self):
        if np.random.random() < self.params["corruption_prob"]:
            return np.random.uniform(-1, 1, size=2)
        try:
            action = expert(self.vehicle, deterministic=True)
            noise = np.random.normal(0, [self.params["steering_noise"], self.params["throttle_noise"]])
            return np.clip(action + noise, -1, 1)
        except Exception:
            return np.random.uniform(-1, 1, size=2)

class NaiveIRLPolicy:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.iteration = 0
        self.params_per_iteration = [
            {"steer_range": 0.8, "throt_range": (0.9, 1)},
            {"steer_range": 0.4, "throt_range": (0.9, 1)},
            {"steer_range": 0.2, "throt_range": (0.9, 1)},
        ]

    def act(self):
        params = self.params_per_iteration[min(self.iteration, len(self.params_per_iteration) - 1)]
        steering = np.random.uniform(-params["steer_range"], params["steer_range"])
        throttle = np.random.uniform(*params["throt_range"])
        return np.array([steering, throttle])

class RobotTeachingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Robot Teaching")
        self.root.configure(bg="#2F4F2F")
        self.center_window()

        self.current_iteration = 0
        self.env = None
        self.policy = None
        self.running = False
        self.current_part = 1
        self.part1_complete = False
        
        if IS_TEACHING_EXPERIMENT:
            self.total_iterations = 6
            self.phase_sequence = ["noisy_expert", "human", "irl", "human", "human_demo", "human_demo"]
            self.part1_iterations = 4
            self.part2_iterations = 2
            self.teaching_seed = 12345
            self.demo_seed = 54321
        else:
            self.total_iterations = 6
            self.phase_sequence = ["human_demo", "human_demo", "noisy_expert", "human", "irl", "human"]
            self.part1_iterations = 2
            self.part2_iterations = 4
            self.demo_seed = 12345
            self.teaching_seed = 54321

        self.trajectory_data = []
        self.current_trajectory = []
        self.session_id = datetime.now().strftime("%m%d_%H%M")
        
        experiment_type = "teaching" if IS_TEACHING_EXPERIMENT else "optimal"
        self.save_dir = os.path.join("trajectory_data", f"{experiment_type}_session_{self.session_id}")
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.teaching_dir = os.path.join(self.save_dir, "teaching")
        self.optimal_dir = os.path.join(self.save_dir, "optimal")
        os.makedirs(self.teaching_dir, exist_ok=True)
        os.makedirs(self.optimal_dir, exist_ok=True)
        self.timestep = 0

        self.setup_ui()

    def center_window(self):
        self.root.update_idletasks()
        screen_w, screen_h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        w, h = 1200, 800
        x, y = (screen_w // 2) - (w // 2), (screen_h // 2) - (h // 2)
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def setup_ui(self):
        main_frame = Frame(self.root, bg="#2F4F2F")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        def create_label(parent, text, font_size, bold=False, side=None, fill=None):
            label = Label(parent, text=text, font=("Arial", font_size, "bold" if bold else "normal"),
                          bg="#2F4F2F", fg="white", justify=tk.CENTER, wraplength=700)
            if side: label.pack(side=side, fill=fill)
            else: label.pack(pady=(0, 15))
            return label

        create_label(main_frame, "Robot Teaching", 24, bold=True)
        status_frame = Frame(main_frame, bg="#2F4F2F")
        status_frame.pack(fill=tk.X, pady=(0, 15))

        self.sim_frame = Frame(main_frame, bg="black", relief=tk.RAISED, borderwidth=3, width=900, height=450)
        self.sim_frame.pack(pady=(0, 15))
        self.sim_frame.pack_propagate(False)
        
        part_name = "TEACH the robot how to drive" if IS_TEACHING_EXPERIMENT else "perform a normal drive"
        self.sim_placeholder = create_label(self.sim_frame, "Please"+part_name+".\n\nClick 'Start Experiment' to begin Part 1", 14)
        self.sim_placeholder.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        button_frame = Frame(main_frame, bg="#2F4F2F")
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
        if is_running: self.sim_placeholder.place_forget()
        else: self.sim_placeholder.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def start_experiment(self):
        self.running = True
        self.current_iteration = 1
        self.update_ui_state(is_running=True)
        policy_type = self.phase_sequence[0]
        self.run_demonstration(policy_type)

    def next_phase(self):
        if self.current_trajectory:
            self.save_trajectory()
        
        if self.current_part == 1 and self.current_iteration >= self.part1_iterations:
            self.complete_part1()
            return
        
        if self.current_part == 2 and (self.current_iteration - self.part1_iterations) >= self.part2_iterations:
            self.complete_experiment()
            return
        
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
        if self.current_trajectory:
            self.save_trajectory()
            
        self.running = False
        if self.env:
            self.env.close()
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

    def complete_experiment(self):
        self.stop_simulation(completed=True)

    def run_demonstration(self, policy_type):
        self.current_trajectory = []
        self.timestep = 0
        self.current_policy_type = policy_type
        self.recording_started = False
        self.start_position = None
        
        if self.env: self.env.close()
        
        if IS_TEACHING_EXPERIMENT:
            seed = self.teaching_seed if self.current_iteration <= 4 else self.demo_seed
        else:
            seed = self.demo_seed if self.current_iteration <= 2 else self.teaching_seed
        
        is_manual = policy_type in ["human", "human_demo"]
        
        config = {"map": "SCS", "traffic_density": 0.1, "num_scenarios": 1, "start_seed": seed,
                  "manual_control": is_manual, "use_render": True, "window_size": (900, 450),
                  "vehicle_config": {"show_navi_mark": True, "show_line_to_navi_mark": True}}
        self.env = MetaDriveEnv(config)
        obs = self.env.reset()[0]

        if policy_type == "noisy_expert": 
            self.policy = NoisyExpertPolicy(self.env.agent)
        elif policy_type == "irl": 
            self.policy = NaiveIRLPolicy(self.env.agent)
            self.policy.iteration = 0
        else: 
            self.policy = None
        
        self.root.after(100, self.center_window)
        self.simulation_loop()

    def simulation_loop(self):
        if not self.running: return
        try:
            if self.current_policy_type in ["human", "human_demo"]:
                obs, reward, terminated, truncated, info = self.env.step(None)
                
                steering = info.get('steering', 0.0) if info else 0.0
                throttle = info.get('acceleration', 0.0) if info else 0.0
                action = [steering, throttle]
            else:
                action = self.policy.act()
                obs, reward, terminated, truncated, info = self.env.step(action)
            
            if self.current_policy_type in ["human", "human_demo"]:
                current_position = np.array([self.env.agent.position[0], self.env.agent.position[1]])
                
                if self.start_position is None:
                    self.start_position = current_position.copy()
                
                if not self.recording_started:
                    distance_from_start = np.linalg.norm(current_position - self.start_position)
                    if distance_from_start >= 0.1:
                        self.recording_started = True
                        self.timestep = 1
                
                if self.recording_started:
                    key_states = self.get_key_states()
                    
                    step_data = {
                        'timestep': self.timestep,
                        'forward': key_states['forward'],
                        'left': key_states['left'],
                        'right': key_states['right'],
                        'brake': key_states['brake'],
                        'vehicle_x': round(self.env.agent.position[0], 5),
                        'vehicle_y': round(self.env.agent.position[1], 5),
                        'vehicle_heading': round(self.env.agent.heading_theta, 5),
                        'vehicle_speed': round(self.env.agent.speed, 5),
                    }
                    
                    self.current_trajectory.append(step_data)
                    self.timestep += 1
            
            if terminated or truncated: 
                self.root.after(500, self.next_phase)
            else: 
                self.root.after(20, self.simulation_loop)
        except Exception as e:
            self.stop_simulation()
    
    def save_trajectory(self):
        if not self.current_trajectory or self.current_policy_type not in ["human", "human_demo"]:
            return
   
        df = pd.DataFrame(self.current_trajectory)
        filename = f"human_trajectory_{self.current_iteration}.csv"
        
        if self.current_policy_type == "human_demo":
            filepath = os.path.join(self.optimal_dir, filename)
        else:
            filepath = os.path.join(self.teaching_dir, filename)
        
        df.to_csv(filepath, index=False)
        self.trajectory_data.extend(self.current_trajectory)
        self.current_trajectory = []

if __name__ == "__main__": 
    root = tk.Tk()
    app = RobotTeachingApp(root)
    root.mainloop()
