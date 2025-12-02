import polars as pl


def time_split_with_gap(data: pl.LazyFrame, test_size: int, gap_size: int = 30):


    test_size = test_size * 24 * 60 * 60 # in sec

    agg = data.select(
        pl.col("timestamp").max().alias("last_ts")
    ).with_columns([
        (pl.col("last_ts") - test_size).alias("test_ts"),
        (pl.col("last_ts") - test_size - gap_size).alias("train_ts"),
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
    


    