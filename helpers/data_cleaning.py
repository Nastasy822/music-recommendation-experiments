import polars as pl


def remove_duplicates_by_timestamps(df_lf: pl.LazyFrame):
    
    keys = ["timestamp", "uid", "event_type"]
    return df_lf.unique(
        subset=keys,
        keep="none",        
        maintain_order=False  
    )


def get_listen_data(lf: pl.LazyFrame):
    return lf.filter(pl.col("event_type") == "listen")


def get_not_listen_data(lf: pl.LazyFrame):
    return lf.filter(pl.col("event_type") != "listen")


def filter_rare_items(df_lf: pl.LazyFrame, min_listens: int = 5):
    return df_lf.filter(
        pl.len().over("item_id") >= min_listens
    )


def filter_rare_users(df_lf: pl.LazyFrame, min_listens: int = 20):
    return df_lf.filter(
        pl.len().over("uid") >= min_listens
    )


def cut_track_len(df: pl.LazyFrame, min_limit = 60, max_limit = 350):
    return df.filter(pl.col("track_length_seconds").is_between(min_limit, max_limit, closed="both")) 


EVENT_MAP = {
    "like": "like",
    "unlike": "dislike",
    "dislike": "dislike",
    "undislike": "like",
    "listen": "listen",
}


def convert_reaction(lazy_df: pl.LazyFrame) -> pl.LazyFrame:
    reaction_events = ["like", "unlike", "dislike", "undislike"]

    return (
        lazy_df
        .filter(pl.col("event_type").is_in(reaction_events))
        .group_by(["uid", "item_id"], maintain_order=True)
        .agg(pl.all().last())
    )
                                                                                                                                                                           
                                                                                                                                                                           
def rename_events(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(pl.col("event_type").replace(EVENT_MAP))


def remove_listened_data(lf):
    lf = (
        lf
        .filter(
            (pl.col("event_type") == "like") |
            (pl.col("played_ratio_pct") >= 50)
        )
        .group_by("uid")
        .agg(pl.col("item_id").alias("items"))
    )

    return lf