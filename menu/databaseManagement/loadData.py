import os
import time

from simple_term_menu import TerminalMenu

from dotenv import load_dotenv
from project.qdrant.app import main
from utils.files.readFiles import readFiles

load_dotenv()

EMBEDDING_DIRECTORY = os.getenv('EMBEDDING_DIRECTORY')

def loadData():
    while True:
        files = readFiles(EMBEDDING_DIRECTORY, extension='.parquet')
        if not files:
            return
        
        fileOptions = files + ["Exit"]

        terminal_menu = TerminalMenu(
            fileOptions,
            title="Embedding Evaluation Menu \n",
            menu_cursor="> ",
            menu_cursor_style=("fg_red", "bold"),
            menu_highlight_style=("standout",),
            clear_screen=True
        )

        selectedIndex = terminal_menu.show()
        if selectedIndex is None or fileOptions[selectedIndex] == "Exit":
            return

        selectedFile = os.path.join(EMBEDDING_DIRECTORY, fileOptions[selectedIndex])

        distancesOptions = ['Cosine Similarity', 'Euclidean Distance']
        terminal_menu = TerminalMenu(
            distancesOptions,
            title="Select Distance Metric \n",
            menu_cursor="> ",
            menu_cursor_style=("fg_red", "bold"),
            menu_highlight_style=("standout",),
            clear_screen=True
        )

        distanceIndex = terminal_menu.show()
        if distanceIndex is None:
            continue

        distance = distancesOptions[distanceIndex]

        executable(distance, selectedFile)

def executable(distance, filePath):
    main(distance, filePath)
