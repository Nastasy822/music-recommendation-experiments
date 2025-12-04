from sklearn.model_selection import train_test_split
from catboost import CatBoostRanker, Pool
import polars as pl
import pandas as pd
import numpy as np

from tqdm import tqdm

from helpers.params_provider import ParamsProvider

from stages.base_stage import BaseStage
from helpers.evaluate import evaluate_model
from models.utils import create_target_last_day

from helpers.candidate_filtration import CandidatesFiltration
from helpers.diversification import DiversificationByArtistAlbum
from helpers.e_greedy_top_k import EGreedyTopK


class HybridModel:
    def __init__(self):

        self.params = ParamsProvider().get_params()

        self.user_id_column =  self.params.base.column_names.user_id
        self.item_id_column =  self.params.base.column_names.item_id
        self.weights_column =  self.params.base.column_names.weights

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



        self.features = pl.scan_parquet(self.params.datasets.features)
        self.train_candidates = pl.scan_parquet(self.params.datasets.train.candidates)

        #чтобы на предсказаниях было побыстрее
        self.test_candidates = pl.scan_parquet(self.params.datasets.test.candidates)
                
        self.test_pd = (
            self.test_candidates.join(
                self.features,
                on=[self.user_id_column, self.item_id_column],
                how="left",
            )
            .collect()
            .fill_null(0)
            .to_pandas()
        )

        
        self.filter_candidates = self.params.HybridModel.filter_candidates
        self.use_diversifier = self.params.HybridModel.use_diversifier
        self.use_e_greedy = self.params.HybridModel.use_e_greedy
        
        self.candidates_filtration = CandidatesFiltration()

        self.train_data_path = self.params.datasets.train.preprocessed
        train_df = pl.scan_parquet(self.train_data_path)

        self.candidates_filtration.fit(train_df)

        self.diversifier = DiversificationByArtistAlbum(
                items_meta_path=self.params.datasets.items_meta,
                max_per_artist=1,
                max_per_album=1,
            )


        self.eg = EGreedyTopK(
                k=10, 
                exploration_rate=0.2, 
                score_col="score", 
                random_state=42)



    def create_dataset(self, train_df):
        target_df = create_target_last_day(train_df)
        # target_df = target_df.set_index([self.user_id_column , self.item_id_column ])
        dataset = self.train_candidates
        # dataset = dataset.set_index([self.user_id_column , self.item_id_column ])
        
        dataset = dataset.join(target_df, on=[self.user_id_column , self.item_id_column ], how="left")
        dataset = dataset.join(self.features, on=[self.user_id_column , self.item_id_column ], how="left") 
        
        dataset = dataset.fill_nan(0)
        
        # dataset = dataset.reset_index()
        return dataset


    def fit(self, train_df):

    
        data = self.create_dataset(train_df)
        
        # удаляем кейсы, где нечего ранжировать
        data = data.filter(
                        pl.col(self.weights_column)
                        .n_unique()
                        .over(self.user_id_column) > 1
                    ).collect().to_pandas()

        data = data.fillna(0)
        


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

    
    def save(self,):
        self.hybrid_model.save_model(self.params.HybridModel.weights_path)

    def load(self,):
        self.hybrid_model.load_model(self.params.HybridModel.weights_path)

    def recommend(self, uid):
        
        candidates_df = self.test_pd[self.test_pd[self.user_id_column] == uid]
        candidates_df = candidates_df.copy()

        if self.filter_candidates:
            candidates_df = self.candidates_filtration.filter(uid, candidates_df)

        scores = self.hybrid_model.predict(candidates_df[self.list_of_features])
        candidates_df["score"] = scores
        candidates_df = candidates_df.sort_values("score", ascending=False)
        
        if self.use_diversifier:
            candidates_df = self.diversifier.diversify(candidates_df)

        if self.use_e_greedy:
            candidates_df = self.eg.apply(candidates_df)
        

        return candidates_df[self.item_id_column ].tolist(), candidates_df["score"].tolist()
