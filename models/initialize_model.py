from models.most_popular import MostPopular
from models.new_items import NewItemsLastNDays
from models.last_listen_recommender import LastListenRecommender
from models.implicit_models_wrapers import ALS, BM25, BPR
from models.item_knn import ItemKNN
from models.random_walk_with_restart import RandomWalkWithRestart
from models.kmeans_embedding import KMeansEmbedding


def initialize_model(architecture_name: str):
    models_config = {
        "MostPopular":              MostPopular,
        "NewItemsLastNDays":        NewItemsLastNDays,
        "LastListenRecommender":    LastListenRecommender,
        "ALS":                      ALS,
        "BM25":                     BM25,
        "ItemKNN":                  ItemKNN,
        "BPR":                      BPR,
        "RandomWalkWithRestart":    RandomWalkWithRestart,
        "KMeansEmbedding":          KMeansEmbedding,
    }

    return models_config[architecture_name]()
