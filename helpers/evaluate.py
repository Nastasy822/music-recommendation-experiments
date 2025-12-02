from tqdm import tqdm
import polars as pl
import numpy as np


def precision_at_k(recommended, relevant, k):
    recommended_at_k = recommended[:k]
    relevant_hits = sum(1 for item in recommended_at_k if item in relevant)
    return relevant_hits / len(recommended_at_k)


def recall_at_k(recommended, relevant, k):
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
    """
    Ожидаемый формат test_lf:
        колонки: "uid", "item_id"
        по одной строке на (uid, item_id)
    """

    # Сужаемся до нужных колонок и загоняем в память
    test_df = test_lf.select(["uid", "item_id"]).collect()

    # Группируем по пользователю и собираем список его item_id
    user_items_df = (
        test_df
        .group_by("uid")
        .agg(pl.col("item_id").unique().alias("items"))
    )

    overall_recall = []
    overall_ndcg = []

    # iter_rows(named=True) даёт dict-подобные строки: {"uid": ..., "items": [...]}
    for row in tqdm(user_items_df.iter_rows(named=True), total=len(user_items_df)):
        uid = row["uid"]
        user_true = set(row["items"])

        if not user_true:
            continue

        rec, weights = model.recommend(uid)

        if not rec:
            continue

        recall = recall_at_k(rec, user_true, k)
        ndcg = ndcg_at_k(rec, user_true, k)

        overall_recall.append(recall)
        overall_ndcg.append(ndcg)

    print(f"Mean Recall@{k}:", np.mean(overall_recall) if overall_recall else 0.0)
    print(f"Mean NDCG@{k}:", np.mean(overall_ndcg) if overall_ndcg else 0.0)