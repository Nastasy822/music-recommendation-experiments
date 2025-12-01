from abc import ABC, abstractmethod
from helpers.params_provider import ParamsProvider
import pickle

class BaseModel(ABC):

    def __init__(self):
        super().__init__()
        self.params = ParamsProvider().get_params()
        self.timestamp_column =  self.params.base.column_names.timestamp
        self.user_id_column =  self.params.base.column_names.user_id
        self.item_id_column =  self.params.base.column_names.item_id
        self.event_type_column =  self.params.base.column_names.event_type
        self.weights_column =  self.params.base.column_names.weights
        self.played_ratio_max_column = self.params.base.column_names.played_ratio_max
        self.listen_count_column = self.params.base.column_names.listen_count
        self.artist_id_column = self.params.base.column_names.artist_id
        self.album_id_column = self.params.base.column_names.album_id

        self.N = self.params.base.max_predicts_count
        

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def recommend(self):
        pass

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)



def load_model(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


