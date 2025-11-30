import polars as pl


def time_split_with_gap(data: pl.LazyFrame, test_size: int, gap_size: int = 30):
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
    


    