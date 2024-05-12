import random
import customtkinter

CANVAS_WIDTH = 992  # pixels
CANVAS_HEIGHT = 704  # pixels

TABLE_WIDTH = 32  # cells
TABLE_HEIGHT = 32  # cells

CELL_WIDTH = int(CANVAS_WIDTH / TABLE_WIDTH)  # pixels per cell
CELL_HEIGHT = int(CANVAS_HEIGHT / TABLE_HEIGHT)  # pixels per cell


def pos_to_table(pos_x, pos_y):
    x_in_table = int(pos_x / CANVAS_WIDTH * TABLE_WIDTH)
    y_in_table = int(pos_y / CANVAS_HEIGHT * TABLE_HEIGHT)

    return x_in_table, y_in_table


class DeutschFrame(customtkinter.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.configure(fg_color="transparent")

        self.canvas = customtkinter.CTkCanvas(self, width=CANVAS_WIDTH, height=CANVAS_HEIGHT)
        self.canvas.grid(row=0, column=0, columnspan=5, sticky="nsew")

        self.bg_color = self.canvas.cget("bg")

        self.table: list[list] = []
        self.table_states: list[list] = []
        for y in range(TABLE_HEIGHT):
            self.table.append([])
            self.table_states.append([])
            for x in range(TABLE_WIDTH):
                self.table_states[y].append(False)
                self.table[y].append(self.canvas.create_rectangle(x * CELL_WIDTH, y * CELL_HEIGHT, (x + 1) * CELL_WIDTH, (y + 1) * CELL_HEIGHT, fill=self.bg_color, outline="blue"))


        self.run_button = customtkinter.CTkButton(self, text="Run", fg_color="blue", hover_color="red", command=self.run_button_callback)
        self.run_button.grid(row=1, column=1, pady=30)

        self.home_button = customtkinter.CTkButton(self, text="Home", fg_color="blue", hover_color="red", command=lambda: self.parent.show_frame("home"))
        self.home_button.grid(row=1, column=0, pady=30)

        self.create_balanced_button = customtkinter.CTkButton(self, text="Balanced", fg_color="blue", hover_color="red", command=self.create_balanced)
        self.create_balanced_button.grid(row=1, column=2, pady=30)

        self.create_constant0_button = customtkinter.CTkButton(self, text="Constant 0", fg_color="blue", hover_color="red", command=lambda: self.create_constant(False))
        self.create_constant0_button.grid(row=1, column=3, pady=30)

        self.create_constant1_button = customtkinter.CTkButton(self, text="Constant 1", fg_color="blue", hover_color="red", command=lambda: self.create_constant(True))
        self.create_constant1_button.grid(row=1, column=4, pady=30)

        self.result_showed = False
        self.result = (0, 0)


    def run_button_callback(self):
        pass

    def create_balanced(self):
        self.create_constant(0)

        lst = []
        for y in range(TABLE_HEIGHT):
            for x in range(TABLE_WIDTH):
                lst.append((x, y))

        for _ in range(TABLE_WIDTH * TABLE_HEIGHT // 2):
            x, y = random.choice(lst)
            self.table_states[y][x] = True
            self.canvas.itemconfig(self.table[y][x], fill="black")

    def create_constant(self, n):
        for y in range(TABLE_HEIGHT):
            for x in range(TABLE_WIDTH):
                self.table_states[y][x] = n
                self.canvas.itemconfig(self.table[y][x], fill="black" if n else self.bg_color)
