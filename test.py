from direct.showbase.ShowBase import ShowBase
from panda3d.core import WindowProperties
import tkinter

class AppTk(ShowBase):

    def __init__(self):
        ShowBase.__init__(self, windowType = 'none')
        
        self.base = base
        self.loader = loader
        self.render = render
        
        self.base.startTk()

        self.frame = self.base.tkRoot
        self.frame.geometry("800x600")
        self.frame.title("Panda")
        self.frame.configure(bg='#2c3e50')  # Dark blue-gray background around 3D window

        # Create a frame to hold the 3D window with padding
        container = tkinter.Frame(self.frame, bg='#34495e', relief='sunken', bd=3)
        container.pack(fill='both', expand=True, padx=20, pady=(20, 10))
        
        # Add text at the bottom
        bottom_text = tkinter.Label(self.frame, text="Test text here", bg='#2c3e50', fg='white', font=('Arial', 12))
        bottom_text.pack(side='bottom', pady=10)

        props = WindowProperties()
        props.set_parent_window(container.winfo_id())
        props.set_origin(0, 0)
        props.set_size(container.winfo_width(), container.winfo_height())

        self.base.make_default_pipe()
        self.base.open_default_window(props = props)

        container.bind("<Configure>", self.resize)

        # Load scene
        scene = self.loader.loadModel("environment")
        if scene:
            scene.reparentTo(self.render)

    def resize(self, event):
        props = WindowProperties()
        props.set_origin(0, 0)
        props.set_size(self.frame.winfo_width(), self.frame.winfo_height())
        self.base.win.requestProperties(props)

app = AppTk()
app.run()
