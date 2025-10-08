import os
import time
import torch
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from utils.logger import logger
from .loadModel import loadModel
from utils.files.models import getModelPath
from utils.files.readParquetFile import readParquetFile

load_dotenv()
log = logger()

MODELS_DATA = os.getenv('MODELS_DATA')
BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE'))

def generateEmbedding(model, texts):
    try:
        if not texts:
            return []
        
        if isinstance(texts, str):
            payload = texts if texts != '' else 'Null'
        else:
            payload = [text if text != '' else 'Null' for text in texts]
        
        embeddings = model.encode(
            payload,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        
        return embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        
    except Exception as e:
        log.error(f"Error generating embeddings: {e}")
        return None
        

def generateEmbeddings(modelName, filePath, storagePath, progressCallback=None):
    start_time = time.time()
    modelPath = getModelPath(MODELS_DATA, modelName)
    log.info(f"Loading model from: {modelPath}")

    model = loadModel(modelPath)
    
    df = readParquetFile(filePath)
    
    col = 'text'
    total_rows = len(df)
    
    texts = df[col].fillna('Null').astype(str)
    texts = texts.replace({'': 'Null'})
    
    all_embeddings = [None] * total_rows
    
    for start_idx in range(0, total_rows, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, total_rows)
        
        batch_texts = texts.iloc[start_idx:end_idx].tolist()
        batch_embeddings = generateEmbedding(model, batch_texts)
        
        if batch_embeddings is not None:
            for i, embedding in enumerate(batch_embeddings):
                all_embeddings[start_idx + i] = embedding
            
            if progressCallback:
                progressCallback(current=end_idx, total=total_rows)
    
    df[f'{col}_vector'] = all_embeddings
    
    dir_path, filename = os.path.split(filePath)
    fileName, fileExt = os.path.splitext(filename)
    newFileName = f'{storagePath}/{fileName}_{modelName}{fileExt}'
    
    df.to_parquet(newFileName, engine='pyarrow', compression='snappy', index=False)
    
    file_size = os.path.getsize(newFileName) / (1024 * 1024)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    log.info(f'Embeddings saved: {newFileName} ({file_size:.2f} MB)')
    log.info(f'Processing time: {elapsed_time:.2f} seconds ({total_rows/elapsed_time:.2f} rows/sec)')