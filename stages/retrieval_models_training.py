import polars as pl
import logging

from helpers.params_provider import ParamsProvider

from stages.base_stage import BaseStage
from helpers.evaluate import evaluate_model
from models.initialize_model import initialize_model

class RetrievalModelsTraining(BaseStage):
    def __init__(self):
        super().__init__()

        self.params = ParamsProvider().get_params()
        self.models = self.params.retrieval_models
  
    def run(self):
        
        train_df = pl.scan_parquet("data/train_df_preprocessed.parquet")

        for model_name, model_path in self.models.items():
            print(model_name)
            model = initialize_model(model_name) 

            model.fit(train_df)
            model.save(model_path)

 
            

