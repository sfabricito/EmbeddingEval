import pandas as pd

def convertCSVtoParquet(input_file, output_file):
    df = pd.read_csv(input_file)
    
    df.to_parquet(output_file, engine='fastparquet')