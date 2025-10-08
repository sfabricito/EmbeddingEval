import re
import json

def loadModels(filePath):
    with open(filePath, 'r') as json_file:
        return json.load(json_file)

def getModelPath(filePath, modelName):
    models = loadModels(filePath)
    print(models)

    for key, value in models.items():
        if key in modelName:
            return value["id"]
    return None


def searchModel(filePath, filename):
    match = re.search(r'_(\d+)_([^_]+)\.parquet', filename)
    models = loadModels(filePath)
    if match:
        return models[match.group(2)]

    return None

def getModelsName(filePath):
    models = loadModels(filePath)
    return [model['name'] for model in models.values()]