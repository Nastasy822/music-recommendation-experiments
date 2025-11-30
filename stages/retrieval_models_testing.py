import logging

import numpy as np
import polars as pl
from utils.models import *
from utils.embeddings import *

from stages.base_stage import BaseStage
from utils.evaluate import evaluate_model

from utils.maps_creater import build_users_history
import json

from box import ConfigBox
from ruamel.yaml import YAML


models_config = {
    "MostPop_by_likes":         MostPop_by_likes,
    "MostPop_by_listen":        MostPop_by_listen,
    "NewItemsLastNDays":        NewItemsLastNDays,
    "LastListenRecommender":    LastListenRecommender,
    "ALS":                      ALS,
    "BM25Rec":                  BM25Rec,
    "ItemKNN":                  ItemKNN,
    "BPR":                      BPR,
    "RandomWalkWithRestart":    RandomWalkWithRestart,
    "CBF_by_embeding_kmean":    CBF_by_embeding_kmean,
}


class RetrievalModels(BaseStage):
    def __init__(self):
        super().__init__()

        yaml = YAML(typ='safe')
        params = ConfigBox(yaml.load(open("params.yaml", encoding='utf-8')))
        self.models = params.retrivals.models


    def run(self):
        
        train_df = pl.scan_parquet("data/train_df_preprocessed.parquet")
        test_df  = pl.scan_parquet("data/test_df_preprocessed_for_eval.parquet") 

        with open("data/item_map.json", "r", encoding="utf-8") as f:
            item_map = json.load(f)

        item_map = {int(k): v for k, v in item_map.items()}

        items_meta = (
            pl.scan_parquet("data/source/items_meta.parquet")
            .with_columns(
                pl.col("item_id").replace(item_map)
            )
            .unique(subset=["item_id"])
            .drop_nulls()
        )

        for model_name in self.models:
            print(model_name)
            model = models_config[model_name]() 

            if model_name == "RandomWalkWithRestart":
                model.fit(train_df, items_meta)
            else:
                model.fit(train_df)

            evaluate_model(model, test_df)
 
            

