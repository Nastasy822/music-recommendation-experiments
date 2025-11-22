import polars as pl


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



def train_test_split(data: pl.LazyFrame, test_size: int, gap_size: int = 30):
    HOUR_SECONDS = 60 * 60
    DAY_SECONDS = 24 * HOUR_SECONDS

    GAP_SIZE = HOUR_SECONDS // 2
    TEST_SIZE = test_size * DAY_SECONDS

    agg = data.select(
        pl.col("timestamp").max().alias("last_ts")
    ).with_columns([
        (pl.col("last_ts") - TEST_SIZE).alias("test_ts"),
        (pl.col("last_ts") - TEST_SIZE - GAP_SIZE).alias("train_ts"),
    ])

    joined = data.join(agg, how="cross")

    train_df = (
        joined
        .filter(pl.col("timestamp") < pl.col("train_ts"))
        .select(data.collect_schema().names())
    )

    test_df = (
        joined
        .filter(pl.col("timestamp") > pl.col("test_ts"))
        .select(data.collect_schema().names())
    )

    return train_df, test_df
    


    