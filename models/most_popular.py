import polars as pl
from models.base_model import BaseModel
import pickle

class MostPopular(BaseModel):

    def __init__(self):
        super().__init__()

        self.rec = []
        self.weights = []
        self.last_days = self.params.MostPopular.last_days
        self.type_event = self.params.MostPopular.type_event

    def fit(self, lf: pl.LazyFrame):

        max_timestamp = lf.select(pl.col(self.timestamp_column).max()).collect().to_series()[0]
        cutoff_ts = max_timestamp - self.last_days * 60 * 60 * 24

        top_df = (
            lf
            .filter(
                (pl.col(self.event_type_column) == self.type_event)
                & (pl.col(self.timestamp_column) > cutoff_ts)
            )
            .group_by(self.item_id_column)
            .agg(pl.len().alias(self.weights_column))
            .sort(self.weights_column, descending=True)
            .collect()
        )

        self.rec = top_df[self.item_id_column].to_list()
        self.weights = top_df[self.weights_column].to_list()


    def recommend(self, uid):
        return self.rec, self.weights