from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
import numpy as np


class ForwardItem(BaseModel):
    '''Входные данные для предсказания'''
    smiles: str = Field(..., description="SMILES строка молекулы")
    features: Dict[str, float] = Field(
        ..., 
        description="Словарь признаков feature1, feature2, ..., feature78",
        min_length=78,
        max_length=78
    )
    
    def get_features_array(self) -> List[float]:
        '''Преобразует словарь признаков в массив'''
        features = []
        for i in range(1, 79):  # 78 признаков
            key = f'feature{i}'
            features.append(float(self.features.get(key, 0.0)))
        return features


class ToxicityResponse(BaseModel):
    '''Ответ с предсказаниями'''
    smiles: str
    acute_toxicity: float = Field(..., ge=0, le=1, description="Вероятность острой токсичности")
    carcinogenicity: float = Field(..., ge=0, le=1, description="Вероятность канцерогенности")
    cardiotoxicity: float = Field(..., ge=0, le=1, description="Вероятность кардиотоксичности")
    dermal_toxicity: float = Field(..., ge=0, le=1, description="Вероятность дермальной токсичности")
    genotoxicity: float = Field(..., ge=0, le=1, description="Вероятность генотоксичности")
    hepatotoxicity: float = Field(..., ge=0, le=1, description="Вероятность гепатотоксичности")
    ocular_toxicity: float = Field(..., ge=0, le=1, description="Вероятность окулярной токсичности")
    oxidative_stress: float = Field(..., ge=0, le=1, description="Вероятность окислительного стресса")
    respiratory_toxicity: float = Field(..., ge=0, le=1, description="Вероятность респираторной токсичности")
    neuro_sensory_toxicity: float = Field(..., ge=0, le=1, description="Вероятность нейросенсорной токсичности")
    immuno_hematotoxicity: float = Field(..., ge=0, le=1, description="Вероятность иммуногематотоксичности")
    reprod_dev_toxicity: float = Field(..., ge=0, le=1, description="Вероятность репродуктивной токсичности")
    endocrine_metabolic_tox: float = Field(..., ge=0, le=1, description="Вероятность эндокринно-метаболической токсичности")
    
    @field_validator('acute_toxicity', 'carcinogenicity', 'cardiotoxicity', 
                    'dermal_toxicity', 'genotoxicity', 'hepatotoxicity',
                    'ocular_toxicity', 'oxidative_stress', 'respiratory_toxicity',
                    'neuro_sensory_toxicity', 'immuno_hematotoxicity',
                    'reprod_dev_toxicity', 'endocrine_metabolic_tox')
    @classmethod
    def round_to_two_decimals(cls, v: float) -> float:
        '''Округляет значения до 2 знаков после запятой'''
        return round(v, 2)


class HealthResponse(BaseModel):
    '''Ответ на health-check'''
    status: str
    service: str
    model_loaded: bool


class HistoryResponse(BaseModel):
    '''Ответ с историей запросов'''
    id: int
    time: str
    endpoint: str
    request_body: Optional[str]
    response_body: Optional[str]
    code: int