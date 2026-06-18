import numpy as np
from typing import List, Dict
from app.models.tox_model import ToxModel
from app.schemas.requests import ForwardItem, ToxicityResponse


class PredictionService:
    
    def __init__(self, model: ToxModel):
        self.model = model
    
    def predict_batch(self, items: List[ForwardItem]) -> List[ToxicityResponse]:
        if not items:
            return []
        
        features_array = np.array([item.get_features_array() for item in items])
        
        predictions = self.model.predict(features_array)
        
        results = []
        for item, pred in zip(items, predictions):
            result = ToxicityResponse(
                smiles=item.smiles,
                acute_toxicity=float(pred[0]),
                carcinogenicity=float(pred[1]),
                cardiotoxicity=float(pred[2]),
                dermal_toxicity=float(pred[3]),
                genotoxicity=float(pred[4]),
                hepatotoxicity=float(pred[5]),
                ocular_toxicity=float(pred[6]),
                oxidative_stress=float(pred[7]),
                respiratory_toxicity=float(pred[8]),
                neuro_sensory_toxicity=float(pred[9]),
                immuno_hematotoxicity=float(pred[10]),
                reprod_dev_toxicity=float(pred[11]),
                endocrine_metabolic_tox=float(pred[12])
            )
            results.append(result)
        
        return results
    
    def predict_single(self, item: ForwardItem) -> ToxicityResponse:
        results = self.predict_batch([item])
        return results[0]