import polars as pl

def codic_of_index(train_lf: pl.LazyFrame, test_lf: pl.LazyFrame):
    # Маппинг пользователей
    user_map_lf = (
        train_lf
        .select(pl.col("uid"))
        .unique()
        .with_row_count("uid_index")   # даём индексам имена-колонки
    )

    # Маппинг айтемов
    item_map_lf = (
        train_lf
        .select(pl.col("item_id"))
        .unique()
        .with_row_count("item_index")
    )

    # Кодируем train: join по uid и item_id, затем заменяем исходные id на индексы
    train_encoded_lf = (
        train_lf
        .join(user_map_lf, on="uid", how="left")
        .join(item_map_lf, on="item_id", how="left")
        .drop(["uid", "item_id"])
        .rename({"uid_index": "uid", "item_index": "item_id"})
    )

    # То же самое для test (uid/item_id которых нет в train → будут null)
    test_encoded_lf = (
        test_lf
        .join(user_map_lf, on="uid", how="left")
        .join(item_map_lf, on="item_id", how="left")
        .drop(["uid", "item_id"])
        .rename({"uid_index": "uid", "item_index": "item_id"})
    )

    # Для item_id_map нам нужен уже материализованный DataFrame
    item_map_df = item_map_lf.collect()
    item_id_map = dict(
        zip(
            item_map_df["item_id"].to_list(),
            item_map_df["item_index"].to_list(),
        )
    )

    return train_encoded_lf, test_encoded_lf, item_id_map



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
        .select(data.columns)
    )

    test_df = (
        joined
        .filter(pl.col("timestamp") > pl.col("test_ts"))
        .select(data.columns)
    )

    return train_df, test_df
    
    