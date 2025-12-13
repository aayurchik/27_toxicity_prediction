from fastapi import FastAPI, HTTPException, Request, status,Header
from fastapi.responses import JSONResponse
from typing import Any, Optional
import json
import numpy as np

import pickle
import json
from pathlib import Path
import numpy as np


# Инициализация приложения FastAPI
app = FastAPI(
    title="FastAPI Toxicity Prediction",
    description="Prediction of toxicity of chemical compounds based on their physicochemical properties",
    version="1.0.0"
)
class Model:
    def  __init__(self, model_path: str = "trained_models/knn_neuro_sensory_toxicity.pkl"):
        self.path = Path(model_path)
        self.model = None
        self.metadata = None
        self.load_model()
    
    def load_model(self):
        try:
            with open(self.path, 'rb') as f:
                self.model = pickle.load(f)
        except FileNotFoundError:
            raise
        except Exception as e:
            raise

    def predict(self, data: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Модель не загружена")
        try:
            return self.model.predict(data)
        except Exception as e:
            raise

class Features:
    def json_parse(data):
        # data = json.loads(json_line)
        smile = data.get('smiles')
        features = []
        for i in range(1,len(data)):  
            feature_key = f'feature{i}'
            if feature_key in data:
                features.append(data[feature_key])
            else:
                features.append(0.0)
        return smile, np.array([features])

model = None

def get_model():
    global model
    if model is None:
        model = Model()
    return model

@app.on_event("startup")
async def startup_event():
    model = get_model()

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to FastAPI Toxicity Prediction",
        "documentation": "/docs",
        "version": "1.0.0"
    }

@app.get("/health", status_code=status.HTTP_200_OK, tags=["Health"])
async def health_check():
    # Используется для проверки работоспособности API
    return {
        "status": "healthy",
        "service": "FastAPI Toxicity Prediction"
    }

# В FastAPI должен быть route типа POST на /forward, который должен принимать один из двух форматов:
# Если данные не содержат изображений, то необходимо передавать входные данные в теле запроса в формате JSON

@app.post("/forward", status_code=status.HTTP_200_OK, tags=["Predict"])
async def forward(request: Request):
    try:
        body = await request.json()
    # Если запрос неверного формата, то должен возвращаться код ошибки 400 с текстом ‘bad request’
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="bad request")
    # аналогично выкидываем если дело пустое
    if body is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="bad request")
    
    model = get_model()
    
    try: 
        results = []
        for i,j in enumerate(body):
            smile, input_data = Features.json_parse(j)
            prediction = model.predict(input_data)
            results.append({
                "smile": smile,
                "toxity": float(prediction[0])
            })
        # Если модель отработала, то возвращаем результаты в одном из подходящих форматов: JSON
        return JSONResponse(content=results)
    # Если модель не смогла выполнить работу, то необходимо вернуть код ошибки 403 и вернуть сообщение: “модель не смогла обработать данные”
    except Exception:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='модель не смогла обработать данные')

# Реализуйте GET-запрос /history, в котором будет показываться история всех запросов. История всех запросов должна находиться в базе данных.   

       
                
