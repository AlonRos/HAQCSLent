import customtkinter

class HomeFrame(customtkinter.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.configure(fg_color="transparent")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure((0, 1), weight=1)

        self.grover_button = customtkinter.CTkButton(self, text="Grover's Algorithm", fg_color="blue", hover_color="red", command=lambda: self.parent.show_frame("grover"))
        self.grover_button.grid(row=0, column=0, pady=(100, 0))

        self.deutsch_button = customtkinter.CTkButton(self, text="Deutsch Algorithm", fg_color="blue", hover_color="red", command=lambda: self.parent.show_frame("deutsch"))
        self.deutsch_button.grid(row=1, column=0, pady=(0, 100))

