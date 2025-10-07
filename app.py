# import curses
# from project.menu.app import menu
# from project.menu.raw_data.app import processRawDataMenu
# from project.menu.embeddings.app import embeddingMenu
# from project.menu.qdrant.app import qdrantMenu

# def run_menu(stdscr):
#     menu(stdscr, 
#          handlers=[processRawDataMenu, embeddingMenu, qdrantMenu],
#          options=["Process Raw Data", "Generate Embeddings", "Qdrant", "Results"], 
#          location="Menu"
#     )

# if __name__ == "__main__":
#     curses.wrapper(run_menu)
