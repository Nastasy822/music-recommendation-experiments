import polars as pl
from helpers.params_provider import ParamsProvider
import json
from helpers.features_functions import *

class FeaturesExtractor:

    def __init__(self):

        self.params = ParamsProvider().get_params()
        self.user_id_column =  self.params.base.column_names.user_id
        self.item_id_column =  self.params.base.column_names.item_id
        self.train_data_path = self.params.datasets.train.preprocessed

        self.features_path = self.params.datasets.features

        self.items_meta_path = self.params.datasets.items_meta

        

    def run(self):

        train_df = pl.scan_parquet(self.train_data_path)

        items_meta = (
            pl.scan_parquet(self.items_meta_path)
            .unique(subset=[self.item_id_column])
            .drop_nulls()
        )


        self.item_features =  (
                            build_item_stats(train_df)
                            .select([ self.item_id_column ] + self.params.HybridModel.item_features )
                            .collect()
                            .to_pandas()
        )
            
        self.user_features =  (
                            user_music_stats(train_df )
                            .select([self.user_id_column] + self.params.HybridModel.user_features)
                            .collect()
                            .to_pandas()
        )
        

        self.item_user_features =  (
                            build_item_user_profile(train_df, items_meta )
                            .select([self.user_id_column , self.item_id_column ,] + self.params.HybridModel.item_user_features)
                            .collect()
                            .to_pandas()
        )
        

        self.item_features.set_index(self.item_id_column , inplace=True)
        self.user_features.set_index(self.user_id_column , inplace=True)
        self.item_user_features.set_index([self.user_id_column , self.item_id_column], inplace=True)
        

        features = self.item_user_features

        features = features.join(self.item_features, on=[self.item_id_column], how="left")
        features = features.join(self.user_features, on=[self.user_id_column], how="left")
        

        features.to_parquet(self.features_path)