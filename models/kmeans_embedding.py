
from helpers.big_data_helper import *
from utils.maps_creater import *

import polars as pl
import faiss
from tqdm import tqdm
import numpy as np
import json

def create_index(filename, item_id_map):
    lf = pl.scan_parquet(filename)
    CHUNK_SIZE = 50_000
    
    # Возьмём размерность D из первого батча
    first_batch = (
        lf
        .slice(0, CHUNK_SIZE)
        .select(["item_id", "normalized_embed"])
        .collect()
        .to_pandas()
    )

    first_batch["item_id"] = first_batch["item_id"].map(item_id_map)
    
    first_batch = first_batch.dropna()    

    
    emb0 = np.vstack(first_batch["normalized_embed"].to_list()).astype("float32")
    N0, D = emb0.shape
    
    index = faiss.IndexFlatIP(D)   # или что-то более сложное, см. ниже
    index.add(emb0)
    
    all_item_ids = []
    all_item_ids.extend(first_batch["item_id"].tolist())
    
    # Теперь итерируемся по файлу батчами
    offset = CHUNK_SIZE
    while True:
        batch = (
            lf
            .slice(offset, CHUNK_SIZE)
            .select(["item_id", "normalized_embed"])
            .collect()
            .to_pandas()
        )
    
        if batch.empty:
            break
    
        batch["item_id"] = batch["item_id"].map(item_id_map)
        batch = batch.dropna()
    
        if batch.empty:
            offset += CHUNK_SIZE
            continue
    
        emb_batch = np.vstack(batch["normalized_embed"].to_list()).astype("float32")
        index.add(emb_batch)
        all_item_ids.extend(batch["item_id"].tolist())
    
        offset += CHUNK_SIZE
        print(f"added up to offset={offset}, total indexed={index.ntotal}")

    return index, all_item_ids


def build_users_history_normal(train_df: pl.LazyFrame | pl.DataFrame):
    hour = 0.5
    decay = 0.9
    tau = 0.0 if hour == 0 else decay ** (1 / 24 / 60 / 60 / (hour / 24))

    train_df = (
        train_df
        .filter((pl.col("played_ratio_pct") >= 100) | (pl.col("event_type") == "like"))
        .filter(pl.col("is_organic") == 1) # ЧТобы опираться именно на пользовательские вкусы
        # .filter(pl.col("event_type") == "listen")
        # максимум timestamp по uid
        .with_columns(
            pl.max("timestamp").over("uid").alias("max_timestamp")
        )
        # "старость" записи
        .with_columns(
            (pl.col("max_timestamp") - pl.col("timestamp")).alias("delta")
        )
        # экспоненциальное затухание, как в примере: tau ** delta
        .with_columns(
            (tau ** pl.col("delta")).alias("weight")
        )
        .group_by(["uid", "item_id"]).agg(pl.sum("weight").alias("conf"))
        .with_columns(
            pl.when(pl.col("conf") < 1e-9).then(0).otherwise(pl.col("conf")).alias("conf")
        )
        .filter(pl.col("conf") > 0)
        .select(["uid", "item_id"])
        .unique()
        .group_by("uid")
        .agg(pl.col("item_id").alias("items"))
        .collect()
        
    )

    # return train_df
    return {
        row["uid"]: set(row["items"])
        for row in train_df.iter_rows(named=True)
    }


def build_users_history_big(train_df: pl.LazyFrame | pl.DataFrame):
    train_df = (
        train_df
        # .filter(pl.col("is_organic") == 1)
        .select(["uid", "item_id"])
        .unique()
        .group_by("uid")
        .agg(pl.col("item_id").alias("items"))
        .collect()
        
    )

    return {
        row["uid"]: set(row["items"])
        for row in train_df.iter_rows(named=True)
    }


def l2_normalize_rows(X, eps=1e-12):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


def spherical_mean_shift(
    X,
    kappa=50.0,          # "узость" ядра: чем больше, тем локальнее
    max_iter=100,
    tol=1e-4,            # критерий сходимости для одной траектории
    merge_angle_deg=5.0  # порог склейки мод в градусах
):

    # 1) Нормируем входные эмбеддинги на сферу
    X_norm = l2_normalize_rows(X)
    n_samples, n_features = X_norm.shape

    # 2) Инициализируем траектории: начинаем из самих точек
    modes = X_norm.copy()

    for it in range(max_iter):
        # Косинусное сходство между текущими точками и всеми данными
        # modes: (n, d), X_norm: (n, d) -> (n, n)
        sim = modes @ X_norm.T

        # vMF-ядро: веса по exp(kappa * cos)
        weights = np.exp(kappa * sim)
        # Нормируем веса по строкам
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)

        # Обновляем точки mean-shift шагом: взвешенное среднее всех X_norm
        new_modes = weights @ X_norm  # (n, d)
        new_modes = l2_normalize_rows(new_modes)

        # Максимальный сдвиг среди всех точек
        shifts = np.linalg.norm(new_modes - modes, axis=1)
        max_shift = shifts.max()

        modes = new_modes

        if max_shift < tol:
            # print(f"Mean-shift converged at iter {it}, max_shift={max_shift}")
            break

    # 3) Кластеризация мод: склеиваем близкие по углу
    # cos(angle) = u·v, angle = arccos(cos)
    merge_angle_rad = np.deg2rad(merge_angle_deg)
    cos_thresh = np.cos(merge_angle_rad)

    labels = -np.ones(n_samples, dtype=int)
    centers = []

    for i in range(n_samples):
        if labels[i] != -1:
            continue  # уже отнесён к какому-то кластеру

        # Новый кластер, стартуем с моды i
        center = modes[i]
        cluster_id = len(centers)

        # Находим все моды, достаточно близкие по углу к этой
        sims = modes @ center  # (n,)
        in_cluster = sims >= cos_thresh

        labels[in_cluster] = cluster_id
        # Можно обновить центр как среднее этих мод (но они и так близки)
        cluster_center = l2_normalize_rows(modes[in_cluster].mean(axis=0, keepdims=True))[0]
        centers.append(cluster_center)

    centers = np.vstack(centers) if centers else np.zeros((0, n_features))

    return labels, centers, modes



class KMeansEmbedding:
    def __init__(self):
        self.k = 10
            
    def fit(self, data):
        
        with open("data/item_map.json", "r", encoding="utf-8") as f:
            item_map = json.load(f)
        
        item_map = {int(k): v for k, v in item_map.items()}

        self.index, self.item_ids = create_index("data/source/filtered_embeddings.parquet", item_map)

        self.data = data
        
        self.id2pos = {iid: i for i, iid in enumerate(self.item_ids)} 

        self.total = []
        self.N = 1000
        
        self.history = build_users_history_normal(self.data)

        self.history_big = build_users_history_big(self.data)

    def spherical_jitter(self, embeddings, level=0.02):
        noise = np.random.normal(0, 1, embeddings.shape)
        noise = noise - (noise * embeddings).sum(axis=1, keepdims=True) * embeddings
        noise = noise / np.linalg.norm(noise, axis=1, keepdims=True)
        emb_noisy = embeddings + level * noise
        emb_noisy = emb_noisy / np.linalg.norm(emb_noisy, axis=1, keepdims=True)
        return emb_noisy
    
        
    def build_user_profile_embed(self, uid):

        if uid in self.history:
            listened_items = self.history[uid]
            indices = [self.id2pos[iid] for iid in listened_items if iid in self.id2pos]
        else:
            return None, None
        
        
        if uid in self.history_big:
            listened_items_big = self.history_big[uid]
            indices_big = [self.id2pos[iid] for iid in listened_items_big if iid in self.id2pos]
        else:
            return None, None
        
        
        if len(indices) <1:
            return None, None
        
        
        vectors = np.vstack([self.index.reconstruct(int(i)) for i in indices]).astype("float32")
        embeddings = np.array(vectors, dtype="float32")
        embeddings = self.spherical_jitter(embeddings, 0.01) # Немного для рандома 
        
        labels, centers, _ =  spherical_mean_shift(
                        embeddings,
                        kappa=5.0,          # "узость" ядра: чем больше, тем локальнее
                        max_iter=30,
                        tol=1e-4,            # критерий сходимости для одной траектории
                        merge_angle_deg=5.0  # порог склейки мод в градусах
                    )

        return centers, indices_big

    def similar_tracks(self, vecs, indices):
    
        # ---- Нормализация входа ----
        if isinstance(vecs, list):
            vecs = np.array(vecs)
    
        
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)

        count = 1
        if int(self.N/len(vecs)) != 0:
            count = int(self.N/len(vecs))

        sims, ids = self.index.search(vecs, count)  # shape: (batch, N)
    
        global_pairs = []
    
        for row_ids, row_sims in zip(ids, sims):
            for pos, score in zip(row_ids, row_sims):
                if pos not in indices:
                    global_pairs.append((self.item_ids[pos], float(score)))

        global_pairs.sort(key=lambda x: x[1], reverse=True)
    
        all_items  = [p[0] for p in global_pairs]
        all_scores = [p[1] for p in global_pairs]
    
        return all_items, all_scores

    def recommend(self, uid):
        vectors, indices = self.build_user_profile_embed(uid)
        if vectors is None:
            return [], []    
        return self.similar_tracks(vectors, indices)
        