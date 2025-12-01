from sklearn.model_selection import train_test_split
from catboost import CatBoostRanker, Pool
import polars as pl
import pandas as pd
import numpy as np

from tqdm import tqdm

from helpers.params_provider import ParamsProvider
from helpers.candidate_generator import CandidateGenerator
from helpers.features_extractor import FeaturesExtracto

from stages.base_stage import BaseStage
from helpers.evaluate import evaluate_model
from models.utils import create_target_last_day
    
    
class HybridModel:
    def __init__(self):

        self.params = ParamsProvider().get_params()

        self.hybrid_model = CatBoostRanker(
                                            iterations=self.params.HybridModel.iterations,
                                            learning_rate=self.params.HybridModel.learning_rate,
                                            depth=self.params.HybridModel.depth,
                                            loss_function=self.params.HybridModel.loss_function,
                                            verbose=self.params.HybridModel.verbose,
                                            task_type=self.params.HybridModel.task_type,
                                            )


        self.list_of_features = [*self.params.HybridModel.user_features, 
                                *self.params.HybridModel.item_features, 
                                *self.params.HybridModel.item_user_features, 
                                *self.params.HybridModel.scores_model]


        self.candidate_model = CandidateGenerator()
        self.features_extractor = FeaturesExtractor()

        self.user_id_column =  self.params.base.column_names.user_id
        self.item_id_column =  self.params.base.column_names.item_id
        self.weights_column =  self.params.base.column_names.weights


    def create_dataset(self, train_df):
        
        hybrid_train_users = train_df.select(self.user_id_column).unique().collect().to_series().to_list()

        dataset = pd.DataFrame()
        for user_id in tqdm(hybrid_train_users[:100]):
            candidates_df =  self.candidate_model.recommend(user_id)
            dataset = pd.concat([dataset, candidates_df])
        
        target_df = create_target_last_day(train_df).collect().to_pandas()

        target_df = target_df.set_index([self.user_id_column , self.item_id_column ])
        dataset = dataset.set_index([self.user_id_column , self.item_id_column ])

        dataset = dataset.join(target_df, on=[self.user_id_column , self.item_id_column ], how="left")
        dataset = dataset.join(self.features, on=[self.user_id_column , self.item_id_column ], how="left") 
        dataset = dataset.fillna(0)

        dataset = dataset.reset_index()
        return dataset


    def fit(self, train_df, items_meta):
        
        self.features = self.features_extractor.get_features(train_df, items_meta)
        data = self.create_dataset(train_df)

        # удаляем кейсе где нечего ранжировать
        data = data.groupby(self.user_id_column ).filter(lambda x: x[self.weights_column].nunique() > 1)
        
        X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
            data[self.list_of_features],
            data[self.weights_column],
            data[self.user_id_column],
            test_size=self.params.HybridModel.test_size,
            random_state=42
        )

        train_df = X_train.copy()
        train_df["label"] = y_train
        train_df["group_id"] = group_train
        
        train_df = train_df.sort_values("group_id").reset_index(drop=True)
        
        X_train_sorted = train_df.drop(columns=["label", "group_id"])
        y_train_sorted = train_df["label"]
        group_train_sorted = train_df["group_id"]
        
        test_df = X_test.copy()
        test_df["label"] = y_test
        test_df["group_id"] = group_test
        
        test_df = test_df.sort_values("group_id").reset_index(drop=True)

        X_test_sorted = test_df.drop(columns=["label", "group_id"])
        y_test_sorted = test_df["label"]
        group_test_sorted = test_df["group_id"]


        train_pool = Pool(
            data=X_train_sorted,
            label=y_train_sorted,
            group_id=group_train_sorted,
        )

        test_pool = Pool(
            data=X_test_sorted,
            label=y_test_sorted,
            group_id=group_test_sorted,
        )

        
        self.hybrid_model.fit(
            train_pool,
            eval_set=test_pool,
        )


    def recommend(self, uid):

        candidates_df =  self.candidate_model.recommend(uid)

        candidates_df = candidates_df.set_index([self.user_id_column , self.item_id_column ])
        candidates_df = candidates_df.join(self.features, how="left")

        candidates_df = candidates_df.fillna(0)
    
        scores = self.hybrid_model.predict(candidates_df[self.list_of_features])
        candidates_df["score"] = scores
        candidates_df = candidates_df.sort_values("score", ascending=False)
        
        return candidates_df.index.get_level_values(self.item_id_column ).tolist(), candidates_df["score"].tolist()
