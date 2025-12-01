import polars as pl
from models.utils import add_exponential_decay
from models.base_model import BaseModel
import pickle

class LastListenRecommender(BaseModel):

    def __init__(self):
        super().__init__()
        
        self.user_fav_songs = None
        self.hour = self.params.LastListenRecommender.hour
        self.decay = self.params.LastListenRecommender.decay

    def fit(self, train_df: pl.DataFrame | pl.LazyFrame):

        tau = 0.0 if self.hour == 0 else self.decay ** (1 / 24 / 60 / 60 / (self.hour / 24))
        
        df_tau = add_exponential_decay(train_df, tau)

        self.user_fav_songs = df_tau.collect()
    

    def recommend(self, uid: int):

        df = (
            self.user_fav_songs
            .filter(pl.col(self.user_id_column) == uid)
            .sort(
                by=[self.weights_column],
                descending=[True],
            )
        )

        rec = df.get_column(self.item_id_column).to_list()
        weights = df.get_column(self.weights_column).to_list()  

        return rec, weights