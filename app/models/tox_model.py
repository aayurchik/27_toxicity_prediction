import torch
import numpy as np
import joblib
from pathlib import Path
from typing import List, Dict, Optional
from app.models.nn_model import StrongToxNet
from app.config import settings


class ToxModel:
    
    TARGET_COLS = [
        'acute_toxicity', 'carcinogenicity', 'cardiotoxicity',
        'dermal_toxicity', 'genotoxicity', 'hepatotoxicity',
        'ocular_toxicity', 'oxidative_stress', 'respiratory_toxicity',
        'neuro_sensory_toxicity', 'immuno_hematotoxicity',
        'reprod_dev_toxicity', 'endocrine_metabolic_tox'
    ]
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        input_size: Optional[int] = None,
        output_size: Optional[int] = None,
        device: str = "cpu"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.input_size = input_size or settings.INPUT_SIZE
        self.output_size = output_size or settings.OUTPUT_SIZE
        self.model_path = model_path or settings.MODEL_PATH
        self.scaler_path = scaler_path or settings.SCALER_PATH
        
        self.model = None
        self.scaler = None
        
        self.load_model()
        self.load_scaler()
    
    def load_model(self):
        try:
            self.model = StrongToxNet(self.input_size, self.output_size).to(self.device)
            
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Модель не найдена: {self.model_path}")
            
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки модели: {str(e)}")
    
    def load_scaler(self):
        try:
            if not Path(self.scaler_path).exists():
                raise FileNotFoundError(f"Scaler не найден: {self.scaler_path}")
            
            self.scaler = joblib.load(self.scaler_path)
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки scaler: {str(e)}")
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Модель не загружена")
        if self.scaler is None:
            raise ValueError("Scaler не загружен")
        
        try:
            scaled_data = self.scaler.transform(data)
            
            tensor_data = torch.tensor(scaled_data, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                logits = self.model(tensor_data)
                probs = torch.sigmoid(logits)
            
            return probs.cpu().numpy()
        
        except Exception as e:
            raise RuntimeError(f"Ошибка предсказания: {str(e)}")
    
    def predict_single(self, features: np.ndarray) -> Dict[str, float]:
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        predictions = self.predict(features)[0]
        
        return {
            col: float(pred)
            for col, pred in zip(self.TARGET_COLS, predictions)
        }
    
    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.scaler is not None