import os
import time
import qdrant_client 
from dotenv import load_dotenv
from utils.logger import logger
from project.qdrant.manageData import manageData
from project.qdrant.createCollection import createCollection
from utils.files.models import searchModel
from utils.tools.updateEnv import updateEnv

load_dotenv()
log = logger()

QDRANT_HOST = os.getenv('QDRANT_HOST')
QDRANT_PORT = os.getenv('QDRANT_PORT')
MODELS_DATA = os.getenv('MODELS_DATA')

def main(distance, filename):
    log.info(f'Executing Program')

    model = searchModel(MODELS_DATA, filename)
    print(model)
    updateEnv('DISTANCE', distance)
    updateEnv('EMBEDDING_MODEL', model['id'])
    updateEnv('VECTOR_SIZE', model['vector_size'])

    log.info(f'Configuring Qdrant Client. Model: {model}, Distance: {distance}, Vector Size: {model["vector_size"]}')
    qdrant = qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    createCollection(qdrant, distance,  model['vector_size'])

    manageData(qdrant, filename)