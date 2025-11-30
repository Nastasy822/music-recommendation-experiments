import polars as pl
import logging
import json

from box import ConfigBox
from ruamel.yaml import YAML

from models.most_popular import MostPopular
from models.new_items import NewItemsLastNDays
from models.last_listen_recommender import LastListenRecommender
from models.implicit_models_wrapers import ALS, BM25, BPR
from models.item_knn import ItemKNN
from models.random_walk_with_restart import RandomWalkWithRestart
from models.kmeans_embedding import KMeansEmbedding


from stages.base_stage import BaseStage
from helpers.evaluate import evaluate_model


models_config = {
    "MostPopular":              MostPopular,
    "NewItemsLastNDays":        NewItemsLastNDays,
    "LastListenRecommender":    LastListenRecommender,
    "ALS":                      ALS,
    "BM25":                     BM25,
    "ItemKNN":                  ItemKNN,
    "BPR":                      BPR,
    "RandomWalkWithRestart":    RandomWalkWithRestart,
    "KMeansEmbedding":          KMeansEmbedding,
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
 
            

