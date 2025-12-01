import polars as pl
from scipy.sparse import coo_matrix
import numpy as np


def add_exponential_decay(train_df: pl.LazyFrame | pl.DataFrame, tau: float):

    # Фильтрация по условиям
    train_df = (
        train_df
        .filter(pl.col("played_ratio_pct") > 50)
        .filter(pl.col("event_type") == "listen")
        # максимум timestamp по uid
        .with_columns(
            pl.max("timestamp").over("uid").alias("max_timestamp")
        )
        # "старость" записи
        .with_columns(
            (pl.col("max_timestamp") - pl.col("timestamp")).alias("delta")
        )
        # экспоненциальное затухание, как в примере: tau ** delta
        .with_columns(
            (tau ** pl.col("delta")).alias("weight")
        )
        .group_by(["uid", "item_id"]).agg(pl.sum("weight").alias("conf"))
        .with_columns(
            pl.when(pl.col("conf") < 1e-9).then(0).otherwise(pl.col("conf")).alias("weights")
        )
    )

    return train_df


def merge_data_by_count(train_df: pl.LazyFrame | pl.DataFrame, last_days = 300):

    max_timestamp = train_df.select(pl.col("timestamp").max()).collect().to_series()[0]
    cutoff_ts = max_timestamp - last_days * 60 * 60 * 24

    train_df = train_df.filter(pl.col("timestamp") > cutoff_ts)
    # 1) Имплицитный сигнал: сколько раз слушали и средний процент прослушивания
    train_df_implicit = (
        train_df
        .filter(pl.col("event_type") == "listen")
        .group_by(["uid", "item_id"])
        .agg([
            pl.col("timestamp").count().alias("listen_count"),
            pl.col("played_ratio_pct").max().alias("played_ratio_max"),
        ])
    )

    # 2) Дизлайки: делаем флаг dislike_flag = 1
    train_df_dislike = (
        train_df
        .filter(pl.col("event_type") == "dislike")
        .select(["uid", "item_id"])                 # только ключи
        .unique()
        .with_columns(pl.lit(1).alias("dislike_flag"))
    )

    # 3) Лайки: делаем флаг like_flag = 1
    train_df_like = (
        train_df
        .filter(pl.col("event_type") == "like")
        .select(["uid", "item_id"])
        .unique()
        .with_columns(pl.lit(1).alias("like_flag"))
    )

    # 4) Full join по uid, item_id с двумя фреймами, без дублей ключей
    train_merge = (
            train_df_implicit
                    .join(train_df_dislike, on=["uid", "item_id"], how="full", coalesce=True)
                    .join(train_df_like,    on=["uid", "item_id"], how="full", coalesce=True)
                    .fill_null(0)                    
        )

    return train_merge


def create_target_last_day(train_df):

    max_timestamp = train_df.select(pl.col("timestamp").max()).collect().item()
    last_day = max_timestamp - 60 * 60 * 24 * 2
    
    listens = (
        train_df
        .filter(pl.col("event_type") == "listen")
        .filter(pl.col("timestamp") > last_day)
        .filter(pl.col("played_ratio_pct") > 50)
        .select(["uid" , "item_id"])
        .unique()
        .with_columns(pl.lit(1).alias("weights"))
    )
    return listens 


# ToDo формулу описать!!! зачем логорифирование и описать смысл коэффициентов почему max  
def calculate_conf(lf: pl.LazyFrame) -> pl.LazyFrame:
    lf = lf.filter(pl.col("played_ratio_max")>50)    
    return lf.with_columns(
        (
             (
                pl.col("listen_count").cast(pl.Float64)
            ).log1p()
        ).alias("weights")
    )



def build_id_maps(train_lf: pl.LazyFrame):
    """
    Собирает map-словарь для uid и item_id из train_lf.
    Возвращает два словаря: user_map и item_map.
    """

    user_map_lf = (
        train_lf
        .select("uid")
        .unique()
        .with_row_count("uid_index")
    )

    item_map_lf = (
        train_lf
        .select("item_id")
        .unique()
        .with_row_count("item_index")
    )

    user_df = user_map_lf.collect()
    item_df = item_map_lf.collect()

    user_map = dict(zip(user_df["uid"].to_list(), user_df["uid_index"].to_list()))
    item_map = dict(zip(item_df["item_id"].to_list(), item_df["item_index"].to_list()))

    return user_map, item_map


def map_with_id_maps(df_lf: pl.LazyFrame, user_map: dict, item_map: dict):
    """
    Принимает LazyFrame + словари маппинга, возвращает LazyFrame
    с заменёнными uid/item_id.
    """
    # превращаем dict → LazyFrame для join (самый быстрый способ)
    user_map_lf = pl.DataFrame({
        "uid": list(user_map.keys()),
        "uid_index": list(user_map.values())
    }).lazy()

    item_map_lf = pl.DataFrame({
        "item_id": list(item_map.keys()),
        "item_index": list(item_map.values())
    }).lazy()

    # маппинг
    encoded_lf = (
        df_lf
        .join(user_map_lf, on="uid", how="left")
        .join(item_map_lf, on="item_id", how="left")
        .drop(["uid", "item_id"])
        .rename({"uid_index": "uid", "item_index": "item_id"})
    )

    return encoded_lf



def build_users_history_normal(train_df: pl.LazyFrame | pl.DataFrame):
    hour = 0.5
    decay = 0.9
    tau = 0.0 if hour == 0 else decay ** (1 / 24 / 60 / 60 / (hour / 24))

    train_df = (
        train_df
        .filter((pl.col("played_ratio_pct") >= 100) | (pl.col("event_type") == "like"))
        .filter(pl.col("is_organic") == 1) # ЧТобы опираться именно на пользовательские вкусы
        # .filter(pl.col("event_type") == "listen")
        # максимум timestamp по uid
        .with_columns(
            pl.max("timestamp").over("uid").alias("max_timestamp")
        )
        # "старость" записи
        .with_columns(
            (pl.col("max_timestamp") - pl.col("timestamp")).alias("delta")
        )
        # экспоненциальное затухание, как в примере: tau ** delta
        .with_columns(
            (tau ** pl.col("delta")).alias("weight")
        )
        .group_by(["uid", "item_id"]).agg(pl.sum("weight").alias("conf"))
        .with_columns(
            pl.when(pl.col("conf") < 1e-9).then(0).otherwise(pl.col("conf")).alias("conf")
        )
        .filter(pl.col("conf") > 0)
        .select(["uid", "item_id"])
        .unique()
        .group_by("uid")
        .agg(pl.col("item_id").alias("items"))
        .collect()
        
    )

    # return train_df
    return {
        row["uid"]: set(row["items"])
        for row in train_df.iter_rows(named=True)
    }
