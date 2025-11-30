import logging
from catboost import CatBoostRanker
import numpy as np
import polars as pl

from stages.base_stage import BaseStage
from utils.retrivel import CandidateGenerator
from utils.ranking import HybridModel
from utils.evaluate import evaluate_model
from utils.maps_creater import build_users_history

import json


class Ranking(BaseStage):
    def __init__(self):
        super().__init__()
        pass

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

        print("CandidateGenerator")
        

        candidate_model = CandidateGenerator()
        candidate_model.fit(train_df, items_meta)

        print("HybridModel")

        hybrid = HybridModel()
        hybrid.fit(train_df, candidate_model, items_meta)


        hybrid.hybrid_model = CatBoostRanker(
            iterations=5000,
            learning_rate=0.01,
            depth=6,
            loss_function="YetiRank",
            verbose=10,
            task_type="GPU",
        )


        hybrid.fit_ranker()


        hybrid.use_filter = False
        evaluate_model(hybrid, test_df.filter(pl.col("uid")<500) , 10)
