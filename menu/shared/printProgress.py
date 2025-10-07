import time

def printProgress(tittle=None, start= 0, current=0, total=0):
    percent = (current / total) * 100
    if tittle:
        print(f"\n {tittle} - ", end="")
    print(f"\rProcessing: {current}/{total} rows ({percent:.1f}%)", end="")
    if current == total:
        print()