from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.nearest_neighbours import BM25Recommender

from models.model_with_matrix import ModelWithMatrix
import pickle


class ALS(ModelWithMatrix):

    def __init__(self):
        super().__init__()
        self.model = AlternatingLeastSquares(factors = self.params.ALS.factors,
                                            regularization = self.params.ALS.regularization,
                                            iterations = self.params.ALS.iterations,
                                            random_state = self.params.ALS.random_state,
                                            calculate_training_loss = self.params.ALS.calculate_training_loss) 
        

class BM25(ModelWithMatrix):

    def __init__(self):
        super().__init__()
        self.model = BM25Recommender(K = self.params.BM25.K, 
                                    K1 = self.params.BM25.K1, 
                                    B = self.params.BM25.B, 
                                    num_threads = self.params.BM25.num_threads)

        
class BPR(ModelWithMatrix):

    def __init__(self):
        super().__init__()
        self.model = BayesianPersonalizedRanking(
                        factors = self.params.BPR.factors,       
                        learning_rate = self.params.BPR.learning_rate,
                        regularization = self.params.BPR.regularization,
                        iterations = self.params.BPR.iterations,
                        random_state = self.params.BPR.random_state,
                    )




