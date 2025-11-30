from tqdm import tqdm
import polars as pl
import numpy as np


def precision_at_k(recommended, relevant, k):
    """
    recommended: список рекомендованных элементов (в порядке убывания релевантности)
    relevant: множество (или список) релевантных элементов
    k: количество топовых рекомендаций, которые учитываем
    """
    recommended_at_k = recommended[:k]
    relevant_hits = sum(1 for item in recommended_at_k if item in relevant)
    return relevant_hits / len(recommended_at_k)


def recall_at_k(recommended, relevant, k):
    """
    recommended: список рекомендованных элементов
    relevant: множество релевантных элементов
    k: количество топовых рекомендаций
    """
    recommended_at_k = recommended[:k]
    relevant_hits = sum(1 for item in recommended_at_k if item in relevant)
    return relevant_hits / len(relevant) if len(relevant) > 0 else 0.0



def dcg_at_k(predicted, actual, k):
    return sum(1 / np.log2(i + 2) for i, item in enumerate(predicted[:k]) if item in actual)


def idcg_at_k(actual, k):
    return sum(1 / np.log2(i + 2) for i in range(min(len(actual), k)))


def ndcg_at_k(predicted, actual, k):
    ideal = idcg_at_k(actual, k)
    if ideal == 0:
        return None
    return dcg_at_k(predicted, actual, k) / ideal


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