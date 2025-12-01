import polars as pl
import pandas as pd


class CandidatesFiltration:
    def __init__(self, min_days_since_last_play: int = 2):

        self.min_days_since_last_play = min_days_since_last_play
        self.user_recent_items: dict = {}  

    def fit(self, train_df: pl.LazyFrame | pl.DataFrame):

        ldf = train_df.lazy() if isinstance(train_df, pl.DataFrame) else train_df

        max_ts = ldf.select(pl.col("timestamp").max().alias("max_ts")).collect()["max_ts"][0]
        two_days_sec = self.min_days_since_last_play * 24 * 3600

        last_listens = (
            ldf
            .filter(pl.col("event_type") == "listen")
            .group_by(["uid", "item_id"])
            .agg([
                pl.col("timestamp").max().alias("last_listen_ts"),
            ])
            .with_columns([
                ((pl.lit(max_ts) - pl.col("last_listen_ts")) / 86400.0)
                .alias("days_since_last_play")
            ])
            .filter(pl.col("days_since_last_play") < self.min_days_since_last_play)
            .select(["uid", "item_id"])
            .collect()
        )

        self.user_recent_items = {}
        for uid, item_id in last_listens.iter_rows():
            self.user_recent_items.setdefault(uid, set()).add(item_id)

    def filter(self, uid, candidates):

        banned = self.user_recent_items.get(uid, set())
        if not banned:
            return candidates  

        if isinstance(candidates, (list, tuple)):
            return [item for item in candidates if item not in banned]

        if isinstance(candidates, pd.DataFrame):
            return candidates[~candidates["item_id"].isin(banned)]

        if isinstance(candidates, pl.DataFrame):
            return candidates.filter(~pl.col("item_id").is_in(list(banned)))

        return candidates