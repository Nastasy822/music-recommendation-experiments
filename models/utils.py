import polars as pl
from scipy.sparse import coo_matrix
import numpy as np
from typing import Dict, Set, Optional, Tuple


def add_exponential_decay(df: pl.LazyFrame | pl.DataFrame, tau: float) -> pl.DataFrame:
    df = (
        df
        .filter((pl.col("played_ratio_pct") > 50) & (pl.col("event_type") == "listen"))
    )

    # Вычисляем max_timestamp отдельно
    max_ts = df.select(pl.col("timestamp").max()).collect().item()

    return (
        df
        .with_columns([
            pl.col("timestamp").cast(pl.Int64),
        ])
        # добавляем max_timestamp как константу для всех строк
        .with_columns([
            pl.max("timestamp").over("uid").alias("max_timestamp")
        ])
        .with_columns([
            (pl.col("max_timestamp") - pl.col("timestamp")).alias("delta"),
            (tau ** (pl.col("max_timestamp") - pl.col("timestamp"))).alias("weight"),
        ])
        .group_by(["uid", "item_id"])
        .agg(pl.sum("weight").alias("conf"))
        .with_columns(
            pl.when(pl.col("conf") < 1e-9).then(0).otherwise(pl.col("conf")).alias("weights")
        )
    )


def merge_data_by_count(df: pl.LazyFrame | pl.DataFrame, last_days: int = 300) -> pl.DataFrame:
    """Фильтрует данные за последние дни и объединяет лайки, дизлайки и прослушивания."""
    max_timestamp = df.select(pl.col("timestamp").max()).collect().item()
    cutoff_ts = max_timestamp - last_days * 60 * 60 * 24

    df = df.filter(pl.col("timestamp") > cutoff_ts)

    listens = (
        df
        .filter(pl.col("event_type") == "listen")
        .group_by(["uid", "item_id"])
        .agg([
            pl.len().alias("listen_count"),
            pl.col("played_ratio_pct").max().alias("played_ratio_max"),
        ])
    )

    dislikes = (
        df
        .filter(pl.col("event_type") == "dislike")
        .select(["uid", "item_id"])
        .unique()
        .with_columns(pl.lit(1).alias("dislike_flag"))
    )

    likes = (
        df
        .filter(pl.col("event_type") == "like")
        .select(["uid", "item_id"])
        .unique()
        .with_columns(pl.lit(1).alias("like_flag"))
    )

    return (
        listens
        .join(dislikes, on=["uid", "item_id"], how="full", coalesce=True)
        .join(likes, on=["uid", "item_id"], how="full", coalesce=True)
        .fill_null(0)
    )


def create_target_last_day(df: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
    """Создаёт целевую переменную для последнего дня."""
    max_timestamp = df.select(pl.col("timestamp").max()).collect().item()
    last_day = max_timestamp - 60 * 60 * 24 * 5

    return (
        df
        .filter(pl.col("event_type") == "listen")
        .filter(pl.col("timestamp") > last_day)
        .filter(pl.col("played_ratio_pct") > 50)
        # .select(["uid", "item_id"])
        # .unique()
        # .with_columns(pl.lit(1).alias("weights"))
        .group_by(["uid", "item_id"])
        .agg(pl.count().log1p().alias("weights"))
    )


def calculate_conf(df: pl.LazyFrame) -> pl.LazyFrame:
    """Рассчитывает веса на основе количества прослушиваний."""
    return (
        df
        .filter(pl.col("played_ratio_max") > 50)
        .with_columns(
            (pl.col("listen_count").cast(pl.Float64).log1p()).alias("weights")
        )
    )


def build_id_maps(df: pl.LazyFrame) -> Tuple[Dict, Dict]:
    """Строит маппинги для пользователей и треков."""
    user_df = df.select("uid").unique().with_row_count("uid_index").collect()
    item_df = df.select("item_id").unique().with_row_count("item_index").collect()

    user_map = dict(zip(user_df["uid"].to_list(), user_df["uid_index"].to_list()))
    item_map = dict(zip(item_df["item_id"].to_list(), item_df["item_index"].to_list()))

    return user_map, item_map


def map_with_id_maps(df: pl.LazyFrame, user_map: Dict, item_map: Dict) -> pl.LazyFrame:
    """Маппинг ID построенных маппингов."""
    user_map_df = pl.DataFrame({"uid": list(user_map.keys()), "uid_index": list(user_map.values())}).lazy()
    item_map_df = pl.DataFrame({"item_id": list(item_map.keys()), "item_index": list(item_map.values())}).lazy()

    return (
        df
        .join(user_map_df, on="uid", how="left")
        .join(item_map_df, on="item_id", how="left")
        .drop(["uid", "item_id"])
        .rename({"uid_index": "uid", "item_index": "item_id"})
    )


def build_user_listened_items(df: pl.LazyFrame | pl.DataFrame) -> Dict[int, Set[int]]:
    """Строит множество прослушанных треков для каждого пользователя."""
    hour = 0.5
    decay = 0.9
    tau = 0.0 if hour == 0 else decay ** (1 / 24 / 60 / 60 / (hour / 24))

    result = (
        df
        .filter((pl.col("played_ratio_pct") >= 100) | (pl.col("event_type") == "like"))
        .filter(pl.col("is_organic") == 1)
        .with_columns([
            pl.col("timestamp").cast(pl.Int64),
        ])
        # добавляем max_timestamp как константу для всех строк
        .with_columns([
            pl.max("timestamp").over("uid").alias("max_timestamp")
        ])
        .with_columns([
            (pl.col("max_timestamp") - pl.col("timestamp")).alias("delta"),
            (tau ** (pl.col("max_timestamp") - pl.col("timestamp"))).alias("weight"),
        ])
        .group_by(["uid", "item_id"])
        .agg(pl.sum("weight").alias("conf"))
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

    return {row["uid"]: set(row["items"]) for row in result.iter_rows(named=True)}
