import polars as pl
from models.utils import add_exponential_decay
from models.base_model import BaseModel


class LastListenRecommender(BaseModel):

    def __init__(self):

        self.user_fav_songs = None
        self.hour = 2
        self.decay = 0.9

    def fit(self, train_df: pl.DataFrame | pl.LazyFrame):

        tau = 0.0 if self.hour == 0 else self.decay ** (1 / 24 / 60 / 60 / (self.hour / 24))
        
        df_tau = add_exponential_decay(train_df, tau)

        self.user_fav_songs = df_tau.collect()
        
    def recommend(self, uid: int):

        df = (
            self.user_fav_songs
            .filter(pl.col("uid") == uid)
            .sort(
                by=["conf"],
                descending=[True],
            )
        )

        rec = df.get_column("item_id").to_list()
        weights = df.get_column("conf").to_list()  

        return rec, weights