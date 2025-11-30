from tqdm import tqdm
from utils.metrics import ndcg_at_k, recall_at_k
import numpy as np
import polars as pl

def filtering_listened_items(rec, weights, listened_items: set, k=None):
    filtered_rec = []
    filtered_weights = []

    for item, weight in zip(rec, weights):
        if item in listened_items:
            continue
        filtered_rec.append(item)
        filtered_weights.append(weight)
        if k is not None and len(filtered_rec) >= k:
            break

    return filtered_rec, filtered_weights
    

def evaluate_model(model, test_lf: pl.LazyFrame, k: int = 10):

    overall_recall = []
    overall_ndcg = []
    test_lf = test_lf.collect()
    for i in tqdm(range(len(test_lf))): 
        uid = test_lf[i, "uid"] 
        user_true = set(test_lf[i, "items"] )

        if not user_true:
            continue

        rec, weights = model.recommend(uid)

        if not rec:
            continue

        recall = recall_at_k(rec, user_true, k)
        ndcg = ndcg_at_k(rec, user_true, k)

        overall_recall.append(recall)
        overall_ndcg.append(ndcg)

    print("Mean Recall@{}:".format(k), np.mean(overall_recall))
    print("Mean NDCG@{}:".format(k), np.mean(overall_ndcg))