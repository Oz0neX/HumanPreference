import tkinter as tk
from metadrive.envs.metadrive_env import MetaDriveEnv

class MetaDriveTkApp:
    def __init__(self, width=900, height=600):
        self.root = tk.Tk()
        self.root.title("MetaDrive Embedded")
        self.root.geometry(f"{width}x{height}")
        self.root.configure(bg="#2c3e50")

        self.container = tk.Frame(self.root, bg="#34495e", relief="sunken", bd=3)
        self.container.pack(fill="both", expand=True, padx=20, pady=20)
        self.root.update_idletasks()

        config = {
            "use_render": True,
            "window_size": (width-40, height-40),
            "parent_window": self.container.winfo_id(),
        }
        self.env = MetaDriveEnv(config)
        self.env.reset()

    def panda_step(self):
        self.env.engine.taskMgr.step()
        self.root.after(10, self.panda_step)

    def run(self):
        self.panda_step()
        self.root.mainloop()

if __name__ == "__main__":
    app = MetaDriveTkApp()
    app.run()