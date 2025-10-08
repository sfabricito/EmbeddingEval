#!/usr/bin/env python3
import os
import re

def readFiles(directory, extension=None):
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory.")
        return []

    files_list = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):
            if extension:
                if file_name.lower().endswith(extension.lower()):
                    files_list.append(file_name)
            else:
                files_list.append(file_name)

    files_list.sort(key=extract_number)
    return files_list

def extract_number(name):
    match = re.search(r'(\d+)', name)
    return int(match.group(1)) if match else float('inf')