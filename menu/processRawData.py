import os
import time

from simple_term_menu import TerminalMenu
from utils.readFiles import readFiles
from dotenv import load_dotenv
from project.process_raw_data.app import processRawData

load_dotenv()

RAW_DATA_DIRECTORY = os.getenv("RAW_DATA_DIRECTORY")

def processRawDataMenu():
    files = readFiles(RAW_DATA_DIRECTORY)
    if not files:
        return
    
    menuOptions = files + ["Exit"]

    terminal_menu = TerminalMenu(
        menuOptions,
        title="Embedding Evaluation Menu \n",
        menu_cursor="> ",
        menu_cursor_style=("fg_red", "bold"),
        menu_highlight_style=("standout",),
        clear_screen=True
    )

    selectedIndex = terminal_menu.show()
    if selectedIndex is None or menuOptions[selectedIndex] == "Exit":
        return

    selectedFile = os.path.join(RAW_DATA_DIRECTORY, menuOptions[selectedIndex])

    while True:
        try:
            print("Embedding Evaluation Menu \n")
            rows = int(input("Enter the number of rows to process: "))
            if rows >= 0:
                break
            else:
                print("Please enter a non-negative number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    executable(selectedFile, rows)


def executable(file_path, rows):
    processRawData(file_path, rows)
    print("File processing completed.")

    time.sleep(2)