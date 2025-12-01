import polars as pl
from models.base_model import BaseModel
import pickle

class NewItemsLastNDays(BaseModel):

    def __init__(self):
        super().__init__()
        
        self.rec = []
        self.weights = []
        self.last_days = self.params.NewItemsLastNDays.last_days


    def fit(self, lf: pl.LazyFrame):

        max_timestamp = lf.select(pl.col(self.timestamp_column).max()).collect().to_series()[0]
        cutoff_ts = max_timestamp - self.last_days * 60 * 60 * 24
        
        history = (
                lf
                .filter(pl.col(self.timestamp_column) < cutoff_ts)
                .select(pl.col(self.item_id_column).unique())
                .collect()[self.item_id_column]
                .to_list()
            )
        
        current = (
                lf
                .filter(pl.col(self.timestamp_column) >= cutoff_ts)
                .select(pl.col(self.item_id_column).unique())
                .collect()[self.item_id_column]
                .to_list()
            )

        new = list(set(current) - set(history))
        
        new_items = lf.filter(pl.col(self.item_id_column).is_in(new))

        vc = (
            new_items
            .group_by(self.item_id_column)
            .agg(pl.len().alias(self.weights_column))
            .sort(self.weights_column, descending=True)
            .collect()
        )
        
        self.rec = vc[self.item_id_column].to_list()
        self.weights = vc[self.weights_column].to_list()


    def recommend(self, uid):
        return self.rec, self.weights