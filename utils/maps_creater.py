import polars as pl
from tqdm import tqdm

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


def build_users_history(lf: pl.LazyFrame, last_days=30) -> dict:
    """
    Возвращает dict: uid -> set(item_id) только за последние 2 месяца.
    """

    history_df = (
        lf
        .filter(pl.col("is_organic") == 1)
        .filter(pl.col("event_type") == "listen")
        .select(["uid", "item_id"])
        .unique()
        .group_by("uid")
        .agg(pl.col("item_id").alias("items"))
        .collect()
    )

    return {
        row["uid"]: set(row["items"])
        for row in history_df.iter_rows(named=True)
    }