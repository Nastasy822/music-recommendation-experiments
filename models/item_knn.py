import polars as pl
from models.utils import add_exponential_decay
from models.utils import map_with_id_maps
from sklearn.utils.sparsefuncs_fast import inplace_csr_row_normalize_l2
from models.utils import build_id_maps
from models.utils import create_user_item_matrix
import numpy as np 

from models.base_model import BaseModel


class ItemKNN(BaseModel):

    def __init__(self):
        self.N = 2000
        self.hour = 2
        self.decay = 0.9


    def fit(self, lf):

        tau = 0.0 if self.hour == 0 else self.decay ** (1 / 24 / 60 / 60 / (self.hour / 24))
        
        self.user_map, self.item_map = build_id_maps(lf)
        self.reverse_item_map = {v: k for k, v in self.item_map.items()}
        
        lf = map_with_id_maps(lf, self.user_map, self.item_map)
        
        lf_tau = add_exponential_decay(lf, tau) 
        lf_simple = add_exponential_decay(lf, 1)

        self.matrix = create_user_item_matrix(lf_simple)
        self.matrix = self.matrix.tocsr().astype(np.double) 

        self.matrix_tau = create_user_item_matrix(lf_tau)
        self.matrix_tau = self.matrix_tau.tocsr().astype(np.double) 

        self.user_embeddings = self.matrix_tau @ self.matrix.T
        
        inplace_csr_row_normalize_l2(self.user_embeddings)
        inplace_csr_row_normalize_l2(self.matrix)
        

    def recommend(self, uid):
        uid = self.user_map.get(int(uid) , None)

        if uid is None:
            return [], []
            
        vec = self.user_embeddings[uid]
        if vec.nnz == 0:      
            return [], []

        scores = (self.matrix.T @ vec.T).toarray().ravel()
        
        idx = np.argsort(-scores) 
        top_k = min(self.N, len(idx)) 
        idx_top = idx[:top_k] 
        scores_top = scores[idx_top] 
        item_ids = [self.reverse_item_map[i] for i in idx_top] 
        
        return item_ids, scores_top