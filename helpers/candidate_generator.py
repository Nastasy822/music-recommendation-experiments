from helpers.params_provider import ParamsProvider
from models.base_model import load_model
import pandas as pd
from typing import Dict, Any

class CandidateGenerator:
    def __init__(self):
        self.params = ParamsProvider().get_params()
        self.models_names = self.params.candidate_generator
        self.user_id_column =  self.params.base.column_names.user_id
        self.item_id_column =  self.params.base.column_names.item_id

        self.models = {}

        for model_name, count in self.models_names.items():
            if count > 0:
                try:
                    model = load_model(f"weights/{model_name}.pkl")
                    self.models[model_name] = {
                        "model": model,
                        "limit": count
                    }
                    print(f"Loaded model: {model_name}, limit: {count}")
                except Exception as e:
                    print(f"Failed to load model {model_name}: {e}")


    def gather_candidates(self, uid: int) -> Dict[int, Dict[str, float]]:
        candidates = {}

        for model_name, model_data in self.models.items():
            rec, weights = model_data["model"].recommend(uid)
            rec, weights = rec[:model_data["limit"]], weights[:model_data["limit"]]

            for item, score in zip(rec, weights):
                candidates.setdefault(item, {})[model_name] = score

        return candidates

    def make_features(self, user_id: int, candidates: Dict[int, Dict[str, float]]) -> list:
        
        rows = []
        for item_id, scores in candidates.items():
            row = {
                self.user_id_column : user_id,
                self.item_id_column: item_id,
            }
            for model_name in self.models:
                row[f"score_{model_name}"] = scores.get(model_name, 0)
            rows.append(row)
        return rows

    def recommend(self, user_id):

        candidates = self.gather_candidates(user_id)
        features = self.make_features(user_id, candidates)

        return pd.DataFrame(features)