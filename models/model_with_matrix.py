from models.utils import merge_data_by_count, calculate_conf
from models.base_model import BaseModel
from sklearn.utils.sparsefuncs_fast import inplace_csr_row_normalize_l2
from models.utils import map_with_id_maps
from models.utils import build_id_maps
import numpy as np
import polars as pl 

from scipy.sparse import coo_matrix


class ModelWithMatrix(BaseModel):

    def __init__(self):
        
        super().__init__()
        self.model = None
        
    
    def fit(self, train_df):

        train_df = merge_data_by_count(train_df)
        train_df = calculate_conf(train_df)

        self.user_map, self.item_map = build_id_maps(train_df)
        self.reverse_item_map = {v: k for k, v in self.item_map.items()}
        
        train_df = map_with_id_maps(train_df, self.user_map, self.item_map)
        
        self.matrix = self.create_user_item_matrix(train_df)

        self.matrix = self.matrix.tocsr().astype(np.double) # The algorithm needs exactly double
        inplace_csr_row_normalize_l2(self.matrix)  

        self.model.fit(self.matrix)


    def create_user_item_matrix(self, lf: pl.LazyFrame):
        df = (
            lf
            .select([
                pl.col(self.user_id_column).cast(pl.Int32).alias(self.user_id_column),
                pl.col(self.item_id_column).cast(pl.Int32).alias(self.item_id_column),
                pl.col(self.weights_column).cast(pl.Int32).alias(self.weights_column),
            ])
            .collect() 
        )

        rows = df[self.user_id_column].to_numpy()
        cols = df[self.item_id_column].to_numpy()
        data = df[self.weights_column].to_numpy()

        matrix = coo_matrix((data, (rows, cols))).tocsr()
        return matrix


    def recommend(self, uid):
        uid = self.user_map.get(int(uid) , None)
        if uid is None:
            return [], []
            
        vector = self.matrix[uid]
        if vector.nnz == 0:   
            return [], []
            
        rec, weights = self.model.recommend(
                                userid=uid,
                                user_items=vector,
                                N=self.N,
                                filter_already_liked_items=False,   
                            )

        orig_ids = [self.reverse_item_map[v] for v in rec]

        return list(orig_ids), list(weights)