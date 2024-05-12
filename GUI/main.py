import customtkinter
from subprocess import Popen, PIPE

import grover
import home
import deutsch

WINDOW_WIDTH = 992  # pixels
WINDOW_HEIGHT = 800  # pixels

INPUT_TO_ALG = ".\\input_to_alg.txt"
OUTPUT_FROM_ALG = ".\\output_from_alg.txt"
PROGRAM_PATH = "..\\QuantumProject\\x64\\Release\\QuantumProject.exe"

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.resizable(False, False)
        self.title("HAQCSLent")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.calculating = False

        self.sub_process = Popen([PROGRAM_PATH], stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True, shell=True)

        self.grover_frame = grover.GroverFrame(self)
        self.grover_frame.grid(row=0, column=0, sticky="nsew")

        self.deutsch_frame = deutsch.DeutschFrame(self)
        self.deutsch_frame.grid(row=0, column=0, sticky="nsew")

        self.home_frame = home.HomeFrame(self)
        self.home_frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("home")

    def on_closing(self):
        self.comm(0)
        self.destroy()

    def comm(self, s):
        print(s, file=self.sub_process.stdin, flush=True)

    def show_frame(self, frame_name):
        if self.calculating:
            return

        if frame_name == "home":
            self.home_frame.tkraise()

        elif frame_name == "grover":
            self.grover_frame.tkraise()

        elif frame_name == "deutsch":
            self.deutsch_frame.tkraise()


if __name__ == "__main__":
    app = App()
    app.mainloop()
