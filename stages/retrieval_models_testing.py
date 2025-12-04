import polars as pl
import logging

from helpers.params_provider import ParamsProvider

from stages.base_stage import BaseStage
from helpers.evaluate import evaluate_model
from models.initialize_model import initialize_model
from models.base_model import load_model

class RetrievalModelsTesting(BaseStage):
    
    def __init__(self):
        super().__init__()

        self.params = ParamsProvider().get_params()
        self.models = self.params.retrieval_models
        self.test_data_path = self.params.datasets.test.preprocessed
        self.train_data_path = self.params.datasets.train.preprocessed
  
    def run(self):
        
        test_df  = pl.scan_parquet(self.test_data_path) 
        train_df  = pl.scan_parquet(self.train_data_path) 

        for model_name, model_path in self.models.items():
            print(model_name)
            model = load_model(model_path) 

            evaluate_model(model, test_df, train_df)
 
            

