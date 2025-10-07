import os
import csv
import time
import random
import curses
import tempfile
from dotenv import load_dotenv
from utils.convertCSVtoParquet import convertCSVtoParquet

load_dotenv()

PROCESSED_DATA_DIRECTORY = os.getenv('PROCESSED_DATA_DIRECTORY')

def openCSV(input_file, output_file, num_rows, progress_callback=None):
    with open(input_file, mode='r', newline='', encoding='ISO-8859-1') as infile:
        reader = list(csv.reader(infile))
        header = reader[0]
        data = reader[1:]
        
        selected_rows = random.sample(data, num_rows)

        with open(output_file, mode='w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow([
                "id", "date", "location", "attack_type", "target_type", "target",
                "orchestrating_group", "motive", "weapon", "deceased", "comments", "text"
            ])
            
            total_rows = len(selected_rows)
            for i, row in enumerate(selected_rows):
                row_to_insert = formatRow(row, i)
                if row_to_insert:
                    writer.writerow(row_to_insert)

                if progress_callback:
                    progress_callback(i + 1, total_rows)
                    time.sleep(0.05) 


def printProgress(current, total):
    percent = (current / total) * 100
    print(f"\rProcessing: {current}/{total} rows ({percent:.1f}%)", end="")
    if current == total:
        print()

def joinNonEmpty(*args):
    return ", ".join(str(arg) for arg in args if arg not in [None, "", " "]).strip()

def cleanText(value):
    return "" if not value or value.strip() == "" or value.strip() == "Unknown" or value == "null" else value

def formatRow(row, id):
    result = [
        id,                                                     # id
        row[16],                                                # date
        joinNonEmpty(row[1], row[0], row[2], row[3], row[4]),   # location
        row[5],                                                 # attack type
        row[6],                                                 # target type
        row[7],                                                 # target
        row[8],                                                 # orchestrating_group
        cleanText(row[9]),                                      # motive (removes empty or space-only text)
        joinNonEmpty(row[10], row[11], row[12]),                # weapon
        row[13],                                                # deceased
        joinNonEmpty(row[14], row[15]),                         # comments
        row[17],                                                # text
    ]

    return result


def processRawData(input_file, rows=10):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    try:
        openCSV(input_file, temp_file.name, rows, progress_callback=printProgress)

        os.makedirs(PROCESSED_DATA_DIRECTORY, exist_ok=True)
        output_file = f"{PROCESSED_DATA_DIRECTORY}/globalTerrorism_{rows}.parquet"

        convertCSVtoParquet(temp_file.name, output_file)
    finally:
        os.remove(temp_file.name)