import os
import time

from dotenv import load_dotenv
from simple_term_menu import TerminalMenu

from utils.readFiles import readFiles
from utils.files.models import getModelsName
from .shared.printProgress import printProgress
from project.embeddingGenerator.app import generateEmbeddings

load_dotenv()

MODELS_DATA = os.getenv('MODELS_DATA')
EMBEDDING_DIRECTORY = os.getenv('EMBEDDING_DIRECTORY')
PROCESSED_DATA_DIRECTORY = os.getenv("PROCESSED_DATA_DIRECTORY")

def embeddingsGeneration():
    while True:
        models = getModelsName(MODELS_DATA)
        if not models:
            return
        
        models = models + ['Exit']

        terminal_menu = TerminalMenu(
            models,
            title="Embedding Evaluation Menu \n",
            menu_cursor="> ",
            menu_cursor_style=("fg_red", "bold"),
            menu_highlight_style=("standout",),
            clear_screen=True
        )

        selectedIndex = terminal_menu.show()
        if selectedIndex is None or models[selectedIndex] == 'Exit':
            return
        
        selectedModel = models[selectedIndex]

        files = readFiles(PROCESSED_DATA_DIRECTORY)
        if not files:
            return
        
        files = files + ["Exit"]

        terminal_menu = TerminalMenu(
            files,
            title="Embedding Evaluation Menu \n",
            menu_cursor="> ",
            menu_cursor_style=("fg_red", "bold"),
            menu_highlight_style=("standout",),
            clear_screen=True
        )

        selectedIndex = terminal_menu.show()
        if selectedIndex is None or files[selectedIndex] == "Exit":
            continue

        selectedFile = os.path.join(PROCESSED_DATA_DIRECTORY, files[selectedIndex])
        executable(selectedModel, selectedFile, storagePath=EMBEDDING_DIRECTORY, callbackProgress=printProgress)

def executable(modelName, filePath, storagePath, callbackProgress=None):
    generateEmbeddings(modelName, filePath, storagePath, progressCallback=callbackProgress)
    time.sleep(2)