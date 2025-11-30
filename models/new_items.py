import polars as pl
from models.base_model import BaseModel

class NewItemsLastNDays(BaseModel):

    def __init__(self):

        self.rec = []
        self.weights = []
        self.last_days = 5

    def fit(self, lf: pl.LazyFrame):

        max_timestamp = lf.select(pl.col("timestamp").max()).collect().to_series()[0]
        cutoff_ts = max_timestamp - self.last_days * 60 * 60* 24
        
        history = (
                lf
                .filter(pl.col("timestamp") < cutoff_ts)
                .select(pl.col("item_id").unique())
                .collect()["item_id"]
                .to_list()
            )
        
        current = (
                lf
                .filter(pl.col("timestamp") >= cutoff_ts)
                .select(pl.col("item_id").unique())
                .collect()["item_id"]
                .to_list()
            )

        new = list(set(current) - set(history))
        
        new_items = lf.filter(pl.col("item_id").is_in(new))

        vc = (
            new_items
            .group_by("item_id")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .collect()
        )
        
        self.rec = vc["item_id"].to_list()
        self.weights = vc["count"].to_list()


    def recommend(self, uid):
        return self.rec, self.weights