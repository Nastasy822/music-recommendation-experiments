from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender
from implicit.bpr import BayesianPersonalizedRanking
import numpy as np

class MostPop_by_likes:
    def __init__(self):
        pass

    def fit(self, df):
        stat = df[df["event_type"] == "like"]["item_id"].value_counts()
        self.rec = stat.index.tolist()
        self.weights = stat.tolist()
        

    def recommend(self, uid, k = 10):
        return self.rec[:k], self.weights[:k]
        
class MostPop_by_listen:
    def __init__(self):
        pass

    def fit(self, df):
        stat = (
                 df[df["event_type"] == "listen"]
                .groupby("item_id")["uid"]
                .nunique()
                .sort_values(ascending=False)
                )

        self.rec = stat.index.tolist()
        self.weights = stat.tolist()
        

    def recommend(self, uid, k = 10):
        return self.rec[:k], self.weights[:k]


class ALS:
    def __init__(self):
        self.model = AlternatingLeastSquares(factors=128, #Размерность скрытых признаков (эмбеддингов)
                                regularization=0.05,
                                iterations=30,
                                alpha=1,
                                random_state=42)

    def fit(self, matrix):
        self.matrix = matrix
        self.model.fit(self.matrix)

    def recommend(self, uid, k = 10):
        uid = int(uid)  #Обязательно принимает int, не конвертирует float 
        
        row = self.matrix[uid]
        if row.nnz == 0:      # nnz = number of non-zero elements
            return [], []
        rec_items, w_rec = self.model.recommend(
                                userid=uid,
                                user_items=self.matrix[uid],
                                N=k,
                                filter_already_liked_items=False,   
                            )
        return list(rec_items), list(w_rec)


class ItemItemRec:
    def __init__(self):
        self.model = ItemItemRecommender(K=200)

    def fit(self, matrix):
        self.matrix = matrix.tocsr().astype(np.double) # Алгоритм просит именно double
        self.model.fit(self.matrix)

    def recommend(self, uid, k = 10):
        uid = int(uid)  #Обязательно принимает int, не конвертирует float 
        
        row = self.matrix[uid]
        if row.nnz == 0:      # nnz = number of non-zero elements
            return [], []
        rec_items, w_rec = self.model.recommend(
                                            uid,
                                            self.matrix[uid],
                                            N=k, 
                                        )
        return list(rec_items), list(w_rec)


class BPR:
    def __init__(self):
        self.model = BayesianPersonalizedRanking(
                            factors=128,       # размер латентного пространства
                            learning_rate=0.1,
                            regularization=0.01,
                            iterations=150,
                            random_state=42,
                        )
    
    def fit(self, matrix):
        self.matrix = matrix.tocsr()
        self.model.fit(self.matrix)

    def recommend(self, uid, k = 10):
        uid = int(uid)  #Обязательно принимает int, не конвертирует float 
        
        row = self.matrix[uid]
        if row.nnz == 0:      # nnz = number of non-zero elements
            return [], []
        rec_items, w_rec = self.model.recommend(
                                userid=uid,
                                user_items=self.matrix[uid],
                                N=k,
                                filter_already_liked_items=False,   
                            )
        return list(rec_items), list(w_rec)
