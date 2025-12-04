from helpers.params_provider import ParamsProvider
from tqdm import tqdm
import polars as pl
import numpy as np
import faiss
import pickle 


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


def cosine_similarity_at_k(rec_ids, true_ids, index, id2pos, k):
    """
    Вычисляет CosSim@K:
    Для каждого item из Rec@K ищет наиболее похожий item из true_ids.
    Затем усредняет значения максимальных cosine similarity.

    rec_ids  — список рекомендованных item_id (любая длина)
    true_ids — множество item_id из теста
    embed_map — dict: item_id -> normalized embedding
    k — ограничение top-K
    """

    if not rec_ids or not true_ids or rec_ids == []:
        return 0.0
    

    # ограничение top-K
    rec_ids = rec_ids[:k]
    
    indices = [id2pos[iid] for iid in true_ids if iid in id2pos]
    
    if indices == []:
        return 0.0

    true_embs = np.vstack([index.reconstruct(int(i)) for i in indices]).astype("float32")

    if true_embs.size == 0:
        return 0.0

    sims = []

    for rid in rec_ids:
        if rid not in id2pos:
            sims.append(0.0)
            continue


        rec_emb = index.reconstruct(int(id2pos[rid]))

        # раз все эмбеддинги нормализованы, cosine = dot product
        cos_vals = true_embs @ rec_emb  # shape (num_true,)

        sims.append(float(np.max(cos_vals)))

    return float(np.mean(sims)) if sims else 0.0


def evaluate_model(
    model,
    test_lf: pl.LazyFrame,
    train_lf: pl.LazyFrame,
    k: int = 10
):
    
    """
    test_lf: содержит "uid", "item_id"
    train_lf: содержит "item_id" (каталог, либо обучающие айтемы)

    Считаем:
        - Recall@k
        - NDCG@k
        - ItemCoverage@k: доля train-айтемов, покрытых рекомендациями
    """

    # --- Подготовка теста ---
    test_df = test_lf.select(["uid", "item_id"]).collect()

    user_items_df = (
        test_df
        .group_by("uid")
        .agg(pl.col("item_id").unique().alias("items"))
    )

    # --- Подготовка train (каталога) ---
    train_items = set(
        train_lf.select("item_id").unique().collect()["item_id"].to_list()
    )

    # --- Индекса с эмбеддингами ---
    embeddings_index_path = ParamsProvider().get_params().datasets.embeddings_index
    index_item_ids_map_path = ParamsProvider().get_params().datasets.index_item_ids_map
    index = faiss.read_index(embeddings_index_path)

    with open(index_item_ids_map_path, "rb") as f:
            item_ids = pickle.load(f)

    id2pos = {iid: i for i, iid in enumerate(item_ids)} 

    # ------------------------------------

    overall_recall = []
    overall_ndcg = []
    overall_cos_sim = []

    recommended_items_at_k = set()

    for row in tqdm(user_items_df.iter_rows(named=True), total=len(user_items_df)):
        uid = row["uid"]
        user_true = set(row["items"])

        if not user_true or user_true == []:
            continue

        rec, weights = model.recommend(uid)

        if not rec:
            continue
        

        # берём топ-k
        rec_k = rec[:k]

        # копим ай̆темы для coverage@test
        recommended_items_at_k.update(rec_k)

        recall = recall_at_k(rec_k, user_true, k)
        ndcg = ndcg_at_k(rec_k, user_true, k)
        cos_sim = cosine_similarity_at_k(rec_k, user_true, index, id2pos, k)

        overall_recall.append(recall)
        overall_ndcg.append(ndcg)
        overall_cos_sim.append(cos_sim)

    # --- Метрики ---
    mean_recall = float(np.mean(overall_recall)) if overall_recall else 0.0
    mean_ndcg = float(np.mean(overall_ndcg)) if overall_ndcg else 0.0
    mean_cos_sim = float(np.mean(overall_cos_sim)) if overall_cos_sim else 0.0

    # Item Coverage@test = пересечение рекомендованных и train
    if train_items:
        item_coverage_test = len(recommended_items_at_k & train_items) / len(train_items)
    else:
        item_coverage_test = 0.0

    # --- Вывод ---
    print(f"Mean Recall@{k}:         {mean_recall:.6f}")
    print(f"Mean NDCG@{k}:           {mean_ndcg:.6f}")
    print(f"Item Coverage@{k}:      {item_coverage_test:.6f}")
    print(f"Mean Cosine Similarity@{k}:      {mean_cos_sim:.6f}")