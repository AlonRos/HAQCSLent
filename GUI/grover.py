import random
import time

import customtkinter
from main import INPUT_TO_ALG, OUTPUT_FROM_ALG

CANVAS_WIDTH = 992  # pixels
CANVAS_HEIGHT = 690  # pixels

TABLE_WIDTH = 32  # cells
TABLE_HEIGHT = 30  # cells

CELL_WIDTH = int(CANVAS_WIDTH / TABLE_WIDTH)  # pixels per cell
CELL_HEIGHT = int(CANVAS_HEIGHT / TABLE_HEIGHT)  # pixels per cell


def pos_to_table(pos_x, pos_y):
    x_in_table = int(pos_x / CANVAS_WIDTH * TABLE_WIDTH)
    y_in_table = int(pos_y / CANVAS_HEIGHT * TABLE_HEIGHT)

    return x_in_table, y_in_table


class GroverFrame(customtkinter.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.configure(fg_color="transparent")

        self.canvas = customtkinter.CTkCanvas(self, width=CANVAS_WIDTH, height=CANVAS_HEIGHT)
        self.canvas.bind('<Button-1>', self.mouse_click_callback)
        self.canvas.grid(row=0, column=0, columnspan=4, sticky="nsew")

        for x in range(TABLE_WIDTH):
            self.canvas.create_line(x * CELL_WIDTH, 0, x * CELL_WIDTH, CANVAS_HEIGHT)

        for y in range(TABLE_HEIGHT):
            self.canvas.create_line(0, y * CELL_HEIGHT, CANVAS_WIDTH, y * CELL_HEIGHT)

        self.bg_color = self.canvas.cget("bg")

        self.table: list[list] = []
        self.table_states: list[list] = []
        for y in range(TABLE_HEIGHT):
            self.table.append([])
            self.table_states.append([])
            for x in range(TABLE_WIDTH):
                self.table_states[y].append(False)
                self.table[y].append(self.canvas.create_rectangle(x * CELL_WIDTH, y * CELL_HEIGHT, (x + 1) * CELL_WIDTH, (y + 1) * CELL_HEIGHT, fill=self.bg_color))


        self.run_button = customtkinter.CTkButton(self, text="Run", fg_color="blue", hover_color="red", command=self.run_button_callback)
        self.run_button.grid(row=1, column=1, pady=30)

        self.home_button = customtkinter.CTkButton(self, text="Home", fg_color="blue", hover_color="red", command=lambda: self.parent.show_frame("home"))
        self.home_button.grid(row=1, column=0, pady=30)

        self.clear_result_button = customtkinter.CTkButton(self, text="Clear Result", fg_color="blue", hover_color="red", command=self.clear_result)
        self.clear_result_button.grid(row=1, column=2, pady=30)

        self.reset_button = customtkinter.CTkButton(self, text="Reset", fg_color="blue", hover_color="red", command=self.reset)
        self.reset_button.grid(row=1, column=3, pady=30)

        self.amount_clicked = 0
        self.result_showed = False
        self.result = (0, 0)


    def reset(self):
        if self.parent.calculating:
            return

        self.clear_result()
        for y in range(TABLE_HEIGHT):
            for x in range(TABLE_WIDTH):
                self.table_states[y][x] = False
                self.canvas.itemconfig(self.table[y][x], fill=self.bg_color)

        self.amount_clicked = 0

    def draw_result(self):
        table_y, table_x = self.result[1], self.result[0]
        self.canvas.itemconfig(self.table[table_y][table_x], fill="blue")
        self.result_showed = True


    def run_button_callback(self):
        if self.parent.calculating:
            return
        self.parent.calculating = True

        with open(OUTPUT_FROM_ALG, "r") as f:
            if not f.readline() in ("result\n", "read\n"):
                return

        self.clear_result()

        with open(INPUT_TO_ALG, "w") as f:
            s = f"{self.amount_clicked}"
            for y in range(TABLE_HEIGHT):
                for x in range(TABLE_WIDTH):
                    if self.table_states[y][x]:
                        s += f"\n{x} {y}"

            f.write(s)

        with open(OUTPUT_FROM_ALG, "w") as f:
            pass

        self.parent.comm(1)  # grover

        while True:
            with open(OUTPUT_FROM_ALG, "r") as f:
                if f.readline() in ("result\n", "read\n"):
                    break

            time.sleep(0.05)

        with open(OUTPUT_FROM_ALG, "r") as f:
            f.readline()
            result_idx = int(f.readline())

            if result_idx < TABLE_WIDTH * TABLE_HEIGHT:
                self.result = (result_idx % TABLE_WIDTH, result_idx // TABLE_WIDTH)
            else:
                self.result = (random.randrange(TABLE_WIDTH), random.randrange(TABLE_HEIGHT))


        self.draw_result()

        self.parent.calculating = False



    def clear_result(self):
        table_y, table_x = self.result[1], self.result[0]

        if not self.table_states[table_y][table_x]:
            self.canvas.itemconfig(self.table[table_y][table_x], fill=self.bg_color)

        else:
            self.canvas.itemconfig(self.table[table_y][table_x], fill="black")

        self.result_showed = False

    def mouse_click_callback(self, e):
        if self.parent.calculating:
            return

        if 0 <= e.x < CANVAS_WIDTH and 0 <= e.y < CANVAS_HEIGHT:
            table_x, table_y = pos_to_table(e.x, e.y)
            if self.result == (table_x, table_y) and self.result_showed:
                return

            if self.table_states[table_y][table_x]:
                self.table_states[table_y][table_x] = False
                self.amount_clicked -= 1
                self.canvas.itemconfig(self.table[table_y][table_x], fill=self.bg_color)

            else:
                self.table_states[table_y][table_x] = True
                self.amount_clicked += 1
                self.canvas.itemconfig(self.table[table_y][table_x], fill="black")
