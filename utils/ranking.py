from catboost import CatBoostRanker
from utils.evaluate import filtering_listened_items

import polars as pl
from datetime import timedelta
from sklearn.model_selection import train_test_split
from catboost import CatBoostRanker, Pool
import pandas as pd
import numpy as np
from utils.sorting import CadidatesFiltration
from utils.features_extractor import *
from tqdm import tqdm

class HybridModel:
    def __init__(self):
      
        self.N = 10
        self.limit = 50
        self.use_filter = True
    
        self.hybrid_model = CatBoostRanker(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            loss_function="YetiRank",
            verbose=50,
        )
        self.liest_of_features = [
                                  "score_trend",
                                  "score_new", 
                                  "score_als",
                                  "score_item_knn",
                                  "score_graph",
                                  "score_embed",
                                  "item_plays_last_5d",	
                                  "item_plays_last_30d",	
                                  "item_age_days",	
                                  "item_trend",
                                  "user_total_plays",
                                  "user_active_days",
                                  "user_unique_tracks",
                                  "median_daily_plays",
                                  "unique_tracks_share",
                                  "days_since_last_play",
                                   "item_plays_for_time_profile", 
                                    "item_morning_share",          
                                    "item_day_share",              
                                    "item_evening_share",          
                                    "item_night_share",            
                                    "item_weekday_share",         
                                    "item_weekend_share",         
                                    "item_avg_listen_hour",        
                                    "item_avg_listen_weekday", 
                                  "user_plays_for_time_profile",
                                        "morning_share", 
                                        "day_share",
                                        "evening_share",
                                        "night_share",
                                        "weekday_share",
                                        "weekend_share",
                                        "avg_listen_hour",
                                        "avg_listen_weekday",
                                      "listen_count",
                                      "played_ratio_max",
                                      "dislike_flag",
                                      "like_flag",
                                     "ui_days_since_last_play",
                                        "artist_listen_count",
                                        "album_listen_count"
                                 ]

        self.fill_values = {

                                  "score_trend" : 0,
                                  "score_new": 0,
                                  "score_als": 0,
                                  "score_item_knn": 0,
                                  "score_graph": 0,
                                  "item_plays_last_5d": 0,	
                                  "item_plays_last_30d": 0,		
                                  "item_age_days": 9999,	
                                  "item_trend": 0,
                                  "user_total_plays": 0,
                                  "user_active_days": 0,
                                  "user_unique_tracks": 0,
                                  "median_daily_plays": 0,
                                  "unique_tracks_share": 0,
                                  "days_since_last_play": 9999,
                                   "item_plays_for_time_profile": 0, 
                                    "item_morning_share": 0,          
                                    "item_day_share": 0,              
                                    "item_evening_share": 0,          
                                    "item_night_share": 0,            
                                    "item_weekday_share": 0,        
                                    "item_weekend_share": 0,         
                                    "item_avg_listen_hour": 0,        
                                    "item_avg_listen_weekday": 0, 
                                  "user_plays_for_time_profile": 0,
                                        "morning_share": 0,
                                        "day_share": 0,
                                        "evening_share": 0,
                                        "night_share": 0,
                                        "weekday_share": 0,
                                        "weekend_share": 0,
                                        "avg_listen_hour": 0,
                                        "avg_listen_weekday": 0,
                                               "listen_count": 0,
                                      "played_ratio_max": 0,
                                      "dislike_flag": 0,
                                      "like_flag": 0,
                                     "ui_days_since_last_play": 9999,
                                        "artist_listen_count": 0,
                                        "album_listen_count": 0,
                                        "target": 0,
                }
        
    def fit(self, train_df, candidate_model, items_meta):
        self.candidate_model = candidate_model

        self.filter_model = CadidatesFiltration(min_days_since_last_play = 2)
        self.filter_model.fit(train_df)

        self.item_features = build_item_stats(train_df).select([
                                                            "item_id",
                                                            "item_plays_last_5d",	
                                                           "item_plays_last_30d",	
                                                           "item_age_days",	
                                                           "item_trend",
                                                                "item_plays_for_time_profile", 
                                                        "item_morning_share",          
                                                        "item_day_share",              
                                                        "item_evening_share",          
                                                        "item_night_share",            
                                                        "item_weekday_share",         
                                                        "item_weekend_share",         
                                                        "item_avg_listen_hour",        
                                                        "item_avg_listen_weekday", 
                                                            # "artist_id", 
                                                            # "album_id"
            
                                                            
                                                            ]).collect().to_pandas()

        self.user_features = user_music_stats(train_df).select([
                                                                 "uid",
                                                              "user_total_plays",
                                                              "user_active_days",
                                                              "user_unique_tracks",
                                                              "median_daily_plays",
                                                              "unique_tracks_share",
                                                              "days_since_last_play",
                                                             "user_plays_for_time_profile",
                                                                "morning_share", 
                                                                "day_share",
                                                                "evening_share",
                                                                "night_share",
                                                                "weekday_share",
                                                                "weekend_share",
                                                                "avg_listen_hour",
                                                                "avg_listen_weekday",
                                                
                                                ]).collect().to_pandas()


        self.item_user_features =  build_item_user_profile(train_df,items_meta ).select([ 
                                                                 "uid",
                                                              "item_id",
                                                              "listen_count",
                                                              "played_ratio_max",
                                                              "dislike_flag",
                                                              "like_flag",
                                                             "ui_days_since_last_play",
                                                                "artist_listen_count",
                                                                "album_listen_count"
                                                ]).collect().to_pandas()

                            
        self.item_features.set_index("item_id", inplace=True)
        self.user_features.set_index("uid", inplace=True)
        self.item_user_features.set_index(["uid", "item_id"], inplace=True)

        
        
        
        print("Prepere data")
        pairs_df = self.create_target_last_day(train_df).collect().to_pandas()

        hybrid_train_users = train_df.select("uid").unique().collect().to_series().to_list()

             
        print("Prepere data for rancin model")
      
        train_df = pd.DataFrame()
        for user_id in tqdm(hybrid_train_users[:500]):

            
            # кандидаты от гибридной системы
            candidates_df = pd.DataFrame(self.candidate_model.make_features(user_id, self.candidate_model.gather_candidates(user_id)))
            
            #Убираем из обучения то что не будет исспользоваться далее
            # candidates_df = self.filter_model.filter(user_id, candidates_df)
            
                # если модель не смогла ничего отдать – пропускаем пользователя
            candidates_df["uid"] = user_id
            train_df = pd.concat([train_df,candidates_df])
          
        print("Train rancin model")

        
        self.train_df = train_df

        self.train_df = self.train_df.merge(pairs_df, on=["uid", "item_id"], how="left")

        self.train_df = self.train_df.merge(self.item_features, on=["item_id"], how="left")
        self.train_df = self.train_df.merge(self.user_features, on=["uid"], how="left")
        self.train_df = self.train_df.merge(self.item_user_features, on=["uid", "item_id"], how="left") 



        self.train_df = self.train_df.fillna(value=self.fill_values)


    def create_target_last_day(self, train_df):
        # Определяем последний день (последние 24 часа)
        max_timestamp = train_df.select(pl.col("timestamp").max()).collect().item()
        last_day = max_timestamp - 60*60*24*2
        
        listens = (
            train_df
            .filter(pl.col("event_type") == "listen")
            .filter(pl.col("timestamp") > last_day)
            .filter(pl.col("played_ratio_pct") > 50)
            .select(["uid", "item_id"])
            .unique()
            .with_columns(pl.lit(1).alias("target"))
        )
        return listens    
        
    def fit_ranker(self):
        
        data = self.train_df.groupby('uid').filter(lambda x: x['target'].nunique() > 1)
        
        X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
            data[self.liest_of_features],
            data["target"],
            data["uid"],
            test_size=0.2,
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
            verbose=100
        )


    def recommend(self, uid):
        candidates = self.candidate_model.gather_candidates(uid)
        candidates_df = pd.DataFrame(self.candidate_model.make_features(uid, candidates))
        candidates_df["uid"] = uid
        
        if self.use_filter:
            candidates_df = self.filter_model.filter(uid, candidates_df)
    
        # Установите мультииндекс
        candidates_df = candidates_df.set_index(["uid", "item_id"])

        candidates_df = candidates_df.join(self.item_features, on="item_id", how="left")
        candidates_df = candidates_df.join(self.user_features, on="uid", how="left")
        candidates_df = candidates_df.join(self.item_user_features, how="left")

        candidates_df = candidates_df.fillna(value=self.fill_values) 
    
        scores = self.hybrid_model.predict(candidates_df[self.liest_of_features])
        candidates_df["score"] = scores
        candidates_df = candidates_df.sort_values("score", ascending=False)
        
        return candidates_df.index.get_level_values("item_id").tolist(), candidates_df["score"].tolist()
