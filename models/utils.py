import polars as pl
from scipy.sparse import coo_matrix
import numpy as np


HOUR_SECONDS = 60 * 60
DAY_SECONDS = 24 * HOUR_SECONDS

def add_exponential_decay(train_df: pl.LazyFrame | pl.DataFrame, tau: float):
    # 0, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.5, 1.0, 2
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
            pl.when(pl.col("conf") < 1e-9).then(0).otherwise(pl.col("conf")).alias("conf")
        )
    )

    return train_df


def create_user_item_matrix(dlf: pl.LazyFrame):
    # сразу выбираем только нужные колонки и кастуем типы
    df = (
        dlf
        .select([
            pl.col("uid").cast(pl.Int32).alias("uid"),
            pl.col("item_id").cast(pl.Int32).alias("item_id"),
            pl.col("conf").cast(pl.Int32).alias("conf"),
        ])
        .collect()  # вот здесь LazyFrame -> DataFrame
    )

    rows = df["uid"].to_numpy()
    cols = df["item_id"].to_numpy()
    data = df["conf"].to_numpy()

    mat = coo_matrix((data, (rows, cols))).tocsr()
    return mat


def merge_data_by_count(train_df: pl.LazyFrame | pl.DataFrame, last_days = 300):

    max_timestamp = train_df.select(pl.col("timestamp").max()).collect().to_series()[0]
    cutoff_ts = max_timestamp - last_days * DAY_SECONDS

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


# ToDo формулу описать!!! зачем логорифирование и описать смысл коэффициентов почему max  
def calculate_conf(lf: pl.LazyFrame) -> pl.LazyFrame:
    lf = lf.filter(pl.col("played_ratio_max")>50)    
    return lf.with_columns(
        (
            # 20.0 * pl.col("like_flag").cast(pl.Float64)
            # - 10.0 * pl.col("dislike_flag").cast(pl.Float64)
             (
                pl.col("listen_count").cast(pl.Float64)
                # * (pl.col("played_ratio_max").cast(pl.Float64) / 100.0)
            ).log1p()
        ).alias("conf")
    )



def build_id_maps(train_lf: pl.LazyFrame):
    """
    Собирает map-словарь для uid и item_id из train_lf.
    Возвращает два словаря: user_map и item_map.
    """
    # map по пользователям
    user_map_lf = (
        train_lf
        .select("uid")
        .unique()
        .with_row_count("uid_index")
    )

    # map по айтемам
    item_map_lf = (
        train_lf
        .select("item_id")
        .unique()
        .with_row_count("item_index")
    )

    # материализация (они маленькие)
    user_df = user_map_lf.collect()
    item_df = item_map_lf.collect()

    user_map = dict(zip(user_df["uid"].to_list(), user_df["uid_index"].to_list()))
    item_map = dict(zip(item_df["item_id"].to_list(), item_df["item_index"].to_list()))

    return user_map, item_map