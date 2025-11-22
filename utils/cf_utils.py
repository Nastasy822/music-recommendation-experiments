import numpy as np
import polars as pl

HOUR_SECONDS = 60 * 60
DAY_SECONDS = 24 * HOUR_SECONDS

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
