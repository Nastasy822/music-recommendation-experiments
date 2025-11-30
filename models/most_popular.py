import polars as pl
from models.base_model import BaseModel

class MostPopular(BaseModel):

    def __init__(self):

        self.rec = []
        self.weights = []
        self.last_days = 5
        self.type_event = "like"

    def fit(self, lf: pl.LazyFrame):

        max_timestamp = lf.select(pl.col("timestamp").max()).collect().to_series()[0]
        cutoff_ts = max_timestamp - self.last_days * 60 * 60 * 24

        top_df = (
            lf
            .filter(
                (pl.col("event_type") == self.type_event)
                & (pl.col("timestamp") > cutoff_ts)
            )
            .group_by("item_id")
            .agg(pl.len().alias("counts"))
            .sort("counts", descending=True)
            .collect()
        )

        self.rec = top_df["item_id"].to_list()
        self.weights = top_df["counts"].to_list()


    def recommend(self, uid):
        return self.rec, self.weights