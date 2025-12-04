import logging
from catboost import CatBoostRanker
import numpy as np
import polars as pl

from helpers.params_provider import ParamsProvider
from stages.base_stage import BaseStage
from models.hybrid import HybridModel
from helpers.evaluate import evaluate_model

import json


class Ranking(BaseStage):

    def __init__(self):
        super().__init__()
        
        self.params = ParamsProvider().get_params()
        self.train_data_path = self.params.datasets.train.preprocessed
        self.test_data_path = self.params.datasets.test.preprocessed

    def run(self):

        train_df = pl.scan_parquet(self.train_data_path)
        test_df  = pl.scan_parquet(self.test_data_path) 

        hybrid = HybridModel()
        # hybrid.fit(train_df)

        hybrid.load()

        evaluate_model(hybrid, test_df, train_df, 100)
