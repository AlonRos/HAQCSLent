import customtkinter
import grover
import home
import deutsch

WINDOW_WIDTH = 992  # pixels
WINDOW_HEIGHT = 800  # pixels

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.resizable(False, False)
        self.title("HAQCSLent")

        self.grover_frame = grover.GroverFrame(self)
        self.grover_frame.grid(row=0, column=0, sticky="nsew")

        self.deutsch_frame = deutsch.DeutschFrame(self)
        self.deutsch_frame.grid(row=0, column=0, sticky="nsew")

        self.home_frame = home.HomeFrame(self)
        self.home_frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("home")



    def show_frame(self, frame_name):
        if frame_name == "home":
            self.home_frame.tkraise()

        elif frame_name == "grover":
            self.grover_frame.tkraise()

        elif frame_name == "deutsch":
            self.deutsch_frame.tkraise()



if __name__ == "__main__":
    app = App()
    app.mainloop()
