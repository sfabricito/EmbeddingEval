import os
import csv
import time
import random
import curses
import tempfile
from dotenv import load_dotenv
from utils.files.convertCSVtoParquet import convertCSVtoParquet

load_dotenv()

PROCESSED_DATA_DIRECTORY = os.getenv('PROCESSED_DATA_DIRECTORY')

def openCSV(input_file, output_file, num_rows, progressCallback=None):
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

                if progressCallback:
                    progressCallback(current=i + 1, total=total_rows)

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


def processRawData(input_file, rows=10, progressCallback=None):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    try:
        openCSV(input_file, temp_file.name, rows, progressCallback=progressCallback)

        os.makedirs(PROCESSED_DATA_DIRECTORY, exist_ok=True)
        output_file = f"{PROCESSED_DATA_DIRECTORY}/globalTerrorism_{rows}.parquet"

        convertCSVtoParquet(temp_file.name, output_file)
    finally:
        os.remove(temp_file.name)