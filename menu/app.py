#!/usr/bin/env python3
import time
from simple_term_menu import TerminalMenu

from .processRawData import processRawDataMenu
from .embeddingsGeneration import embeddingsGeneration



class Prompt:
    @staticmethod
    def menu(options, title="Main Menu"):
        terminal_menu = TerminalMenu(
            options,
            title=f"{title}\n",
            menu_cursor="> ",
            menu_cursor_style=("fg_red", "bold"),
            menu_highlight_style=("standout",),
            cycle_cursor=True,
            clear_screen=True
        )
        menu_entry_index = terminal_menu.show()
        if menu_entry_index is None:
            return None
        return options[menu_entry_index]

    @staticmethod
    def dict_menu(dict_options, title=""):
        while True:
            selection = Prompt.menu(list(dict_options.keys()), title)
            if selection is None or selection.lower() == "exit":
                print("\nExiting menu...")
                break
            selected_function = dict_options.get(selection)
            if callable(selected_function):
                selected_function()
            else:
                print("Invalid selection.")


# === Actions === #

def Database():
    print("Opening vector database... (connecting to Qdrant or MariaDB, etc.)")
    time.sleep(2)

def generate_charts():
    print("Generating charts... (e.g., Matplotlib or Plotly)")
    time.sleep(2)


# === Main Menu === #
def main():
    options = {
        "Process Raw Data": processRawDataMenu,
        "Generate Embeddings": embeddingsGeneration,
        "Database": Database,
        "Generate Charts": generate_charts,
        "Exit": None
    }
    Prompt.dict_menu(options, title="Embedding Evaluation Menu")


if __name__ == "__main__":
    main()