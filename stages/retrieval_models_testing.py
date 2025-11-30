import logging

import numpy as np
import polars as pl
from utils.models import *
from utils.embeddings import *

from stages.base_stage import BaseStage
from utils.evaluate import evaluate_model

from utils.maps_creater import build_users_history
import json


class RetrievalModels(BaseStage):
    def __init__(self):
        super().__init__()
        pass

    def run(self):

        train_df = pl.scan_parquet("data/train_df_preprocessed.parquet")
        test_df  = pl.scan_parquet("data/test_df_preprocessed_for_eval.parquet") 
        history =  pl.scan_parquet("data/train_encoded_lf.parquet")
        users_history = build_users_history(history, last_days=30)



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

        print("MostPop_by_likes")
        model = MostPop_by_likes()
        model.fit(train_df)
        evaluate_model(model, users_history, test_df)

        print("MostPop_by_listen")
        model = MostPop_by_listen()
        model.fit(train_df)
        evaluate_model(model, users_history, test_df)

        print("NewItemsLastNDays")
        model = NewItemsLastNDays()
        model.fit(train_df)
        evaluate_model(model, users_history, test_df)

        print("LastListenRecommender")
        model = LastListenRecommender()
        model.fit(train_df)
        evaluate_model(model, users_history, test_df, 50) # 50 потому что в планах фильтровать последние 50 прослушанных 

        print("ALS")
        model = ALS()
        model.fit(train_df)
        evaluate_model(model, users_history, test_df)

        print("BM25Rec")
        model = BM25Rec()
        model.fit(train_df)
        evaluate_model(model, users_history, test_df)

        print("ItemKNN")
        model = ItemKNN()
        model.fit(train_df)
        evaluate_model(model, users_history, test_df, k=10)

        print("BPR")
        model = BPR()
        model.fit(train_df)
        evaluate_model(model, users_history, test_df)


        print("BPR")
        model = BPR()
        model.fit(train_df)
        evaluate_model(model, users_history, test_df)
      

        print("RandomWalkWithRestart")
        model = RandomWalkWithRestart()
        model.fit(train_df, items_meta)
        evaluate_model(model, users_history, test_df)


        print("CBF_by_embeding_kmean")
        model = CBF_by_embeding_kmean()
        model.fit(train_df)
        evaluate_model(model, users_history, test_df , 10)