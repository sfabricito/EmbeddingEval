import os
import time
import subprocess
import threading

from simple_term_menu import TerminalMenu

from .loadData import loadData
from utils.logger import logger
from utils.instances.isHostRunning import isHostRunning

# from .searchMenu import searchMenu, searchParamsMenu, processAllQueries
# from .distanceMenu import distanceMenu

log = logger()

class DatabaseManager:
    def __init__(self):
        self.databaseRunning = False
        self.statusThread = None
        self._stopStatusCheck = threading.Event()
        
    def checkDatabaseStatus(self):
        while not self._stopStatusCheck.is_set():
            new_status = isHostRunning("127.0.0.1", 6333)
            if new_status != self.databaseRunning:
                self.databaseRunning = new_status
            time.sleep(0.5)
    
    def startStatusMonitor(self):
        self._stopStatusCheck.clear()
        self.databaseRunning = isHostRunning("127.0.0.1", 6333)
        self.statusThread = threading.Thread(target=self.checkDatabaseStatus)
        self.statusThread.daemon = True
        self.statusThread.start()
    
    def stopStatusMonitor(self):
        self._stopStatusCheck.set()
        if self.statusThread and self.statusThread.is_alive():
            self.statusThread.join(timeout=1.0)

class DatabaseManagement:
    def __init__(self):
        self.manager = DatabaseManager()
        self.menu_style = {
            "menu_cursor": "> ",
            "menu_cursor_style": ("fg_red", "bold"),
            "menu_highlight_style": ("standout",),
            "clear_screen": True
        }
    
    def _runScript(self, scriptName: str, wait_time: int = 0):
        scriptPath = f"utils/scripts/{scriptName}"
        log.info(f"Executing: {scriptName}")
        try:
            subprocess.run(["bash", scriptPath], check=True)
            if wait_time > 0:
                time.sleep(wait_time)
        except subprocess.CalledProcessError as e:
            log.error(f"Failed to execute {scriptName}: {e}")
    
    def _create_menu(self, title: str, options: list) -> TerminalMenu:
        return TerminalMenu(
            options,
            title=title,
            **self.menu_style
        )
    
    def _handleDatabaseOperation(self, option: str) -> bool:
        operations = {
            "Start": lambda: self._runScript("startQdrant.sh", 5),
            "Stop": lambda: self._runScript("stopQdrant.sh"),
            "Restart": lambda: self._runScript("restartQdrant.sh"),
            "Load Data": loadData,
            # "Search all queries": processAllQueries
        }
        
        if option in operations:
            if option in ["Start", "Stop", "Restart"]:
                operations[option]()
            else:
                operations[option]()
            return True
        elif option == "Back":
            return False
        
        return True
    
    def show(self):
        self.manager.startStatusMonitor()
        last_status = None 
        
        try:
            while True:
                current_status = self.manager.databaseRunning
                if current_status != last_status:
                    os.system("clear")
                    status_text = "Running" if current_status else "Not Running"
                    print(f"Database Status changed: {status_text}")
                    last_status = current_status

                title = f"Database Status: {'Running' if current_status else 'Not Running'}\nUse arrow keys to navigate. Press ENTER to select.\n"
                options = (
                    ["Stop", "Restart", "Load Data", "Search all queries", "Back"]
                    if current_status else
                    ["Start", "Back"]
                )

                terminal_menu = self._create_menu(title, options)
                selected_index = terminal_menu.show()

                if selected_index is None:
                    break

                selected_option = options[selected_index]
                if not self._handleDatabaseOperation(selected_option):
                    break

        except KeyboardInterrupt:
            log.info("Menu interrupted by user")
        finally:
            self.manager.stopStatusMonitor()

def databaseManagementMenu():
    menu = DatabaseManagement()
    menu.show()