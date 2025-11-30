from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.nearest_neighbours import BM25Recommender

from models.model_with_matrix import ModelWithMatrix


class ALS(ModelWithMatrix):

    def __init__(self):
        super().__init__()
        self.model = AlternatingLeastSquares(factors=128,
                                            regularization=0.001,
                                            iterations=15,
                                            random_state=42,
                                            calculate_training_loss = True) 
        

class BM25(ModelWithMatrix):

    def __init__(self):
        super().__init__()
        self.model = BM25Recommender(K=200, 
                                    K1=0.1, 
                                    B=0.75, 
                                    num_threads=0)

        

class BPR(ModelWithMatrix):

    def __init__(self):
        super().__init__()
        self.model = BayesianPersonalizedRanking(
                        factors=64,       
                        learning_rate=0.1,
                        regularization=0.01,
                        iterations=20,
                        random_state=42,
                    )




