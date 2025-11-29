import logging

import numpy as np
import polars as pl

from stages.base_stage import BaseStage
from utils.big_data_helper import estimate_parquet_ram_usage, apply_function_by_batch, concat_files
from utils.data_preprocess import train_test_split, map_with_id_maps
from utils.data_cleaning import *
from utils.maps_creater import build_id_maps
import json


class DataPreprocessing(BaseStage):
    def __init__(self):
        super().__init__()
        pass

    def run(self):

        data = pl.scan_parquet("data/source/multi_event.parquet")
        estimate_parquet_ram_usage("data/source/multi_event.parquet")

        print("train_test_split")
        train_lf, test_lf = train_test_split(data, 1, gap_size = 30)
        train_lf.sink_parquet("data/train_df.parquet")
        test_lf.sink_parquet("data/test_df.parquet")

        estimate_parquet_ram_usage("data/train_df.parquet")
        estimate_parquet_ram_usage("data/test_df.parquet")

        train_lf = pl.scan_parquet("data/train_df.parquet")
        test_lf = pl.scan_parquet("data/test_df.parquet")

        print("создаём словари индексов")
        user_map, item_map = build_id_maps(train_lf)

        # мапим train/test
        train_encoded_lf = map_with_id_maps(train_lf, user_map, item_map)
        test_encoded_lf  = map_with_id_maps(test_lf,  user_map, item_map)

        train_encoded_lf.sink_parquet("data/train_encoded_lf.parquet")
        test_encoded_lf.sink_parquet("data/test_encoded_lf.parquet")

        
        with open("data/item_map.json", "w", encoding="utf-8") as f:
            json.dump(item_map, f, ensure_ascii=False, indent=2)

        with open("data/user_map.json", "w", encoding="utf-8") as f:
            json.dump(user_map, f, ensure_ascii=False, indent=2)


        print("Берем данные")

        apply_function_by_batch("data/train_encoded_lf.parquet", 
                         "data/train_df_listen.parquet", 
                         get_listen_data, 
                         "timestamp", 
                         batch_size = 10_000_000)

        apply_function_by_batch("data/test_encoded_lf.parquet", 
                                "data/test_df_listen.parquet", 
                                get_listen_data , 
                                "timestamp", 
                                batch_size = 10_000_000)

        apply_function_by_batch("data/train_encoded_lf.parquet", 
                                "data/train_df_likes.parquet", 
                                get_not_listen_data, 
                                "timestamp", 
                                batch_size = 10_000_000)

        apply_function_by_batch("data/test_encoded_lf.parquet", 
                                "data/test_df_likes.parquet", 
                                get_not_listen_data , 
                                "timestamp", 
                                batch_size = 10_000_000)


        print("train_df_listen")

                # Из трейна remove_duplicates_by_timestamps
        apply_function_by_batch("data/train_df_listen.parquet", 
                                "data/train_df_listen_1.parquet", 
                                remove_duplicates_by_timestamps , 
                                "timestamp", 
                                batch_size = 10_000_000)

        # Из трейна filter_rare_items
        apply_function_by_batch("data/train_df_listen_1.parquet", 
                                "data/train_df_listen_2.parquet", 
                                filter_rare_items , 
                                "item_id", 
                                batch_size = 10_000_000)

        # Из трейна filter_rare_users
        apply_function_by_batch("data/train_df_listen_2.parquet", 
                                "data/train_df_listen_3.parquet", 
                                filter_rare_users , 
                                "uid", 
                                batch_size = 10_000_000)

        # Из трейна cut_track_len
        apply_function_by_batch("data/train_df_listen_3.parquet", 
                                "data/train_df_listen_4.parquet", 
                                cut_track_len , 
                                "timestamp", 
                                batch_size = 10_000_000)

        
        print("train_df_likes")


                # Из трейна convert_reaction
        apply_function_by_batch("data/train_df_likes.parquet", 
                                "data/train_df_likes_1.parquet", 
                                convert_reaction , 
                                "uid", 
                                batch_size = 10_000_000)

        # Из трейна rename_events
        apply_function_by_batch("data/train_df_likes_1.parquet", 
                                "data/train_df_likes_2.parquet", 
                                rename_events , 
                                "timestamp", 
                                batch_size = 10_000_000)


        print("test_df_likes")

                # Из теста convert_reaction
        apply_function_by_batch("data/test_df_likes.parquet", 
                                "data/test_df_likes_1.parquet", 
                                convert_reaction , 
                                "uid", 
                                batch_size = 10_000_000)

        # Из теста rename_events
        apply_function_by_batch("data/test_df_likes_1.parquet", 
                                "data/test_df_likes_2.parquet", 
                                rename_events , 
                                "timestamp", 
                                batch_size = 10_000_000)

        

        print("concat_files")

        concat_files("data/test_df_listen.parquet", "data/test_df_likes_2.parquet", "data/test_df_preprocessed.parquet")

        concat_files("data/train_df_listen_4.parquet", "data/train_df_likes_2.parquet", "data/train_df_preprocessed.parquet")

        
        estimate_parquet_ram_usage("data/train_df_preprocessed.parquet")
        estimate_parquet_ram_usage("data/test_df_preprocessed.parquet")


        print("Береме данные до препроцессинка (но закодированные)")
        test_df = pl.scan_parquet("data/test_df_preprocessed.parquet")
        test_users_items_df = remove_listened_data(test_df)

        test_users_items_df.sink_parquet("data/test_df_preprocessed_for_eval.parquet")
        
        print("End")