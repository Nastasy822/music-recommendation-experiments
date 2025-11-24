from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender
from implicit.bpr import BayesianPersonalizedRanking
import numpy as np
import pandas as pd
import polars as pl
import faiss
from tqdm import tqdm
from scipy.sparse import coo_matrix
from utils.maps_creater import *
from utils.data_preprocess import *
from implicit.nearest_neighbours import ItemItemRecommender
from implicit.nearest_neighbours import bm25_weight  # или tfidf_weight
from implicit.nearest_neighbours import CosineRecommender

import numpy as np
import scipy.sparse as sp
import polars as pl
from implicit.nearest_neighbours import BM25Recommender


HOUR_SECONDS = 60 * 60
DAY_SECONDS = 24 * HOUR_SECONDS


class MostPop_by_listen:
    def __init__(self):
        self.max_limit = 2_000
        self.rec = []
        self.weights = []
        self.last_days = 5

    def fit(self, lf: pl.LazyFrame):
        max_timestamp = lf.select(pl.col("timestamp").max()).collect().to_series()[0]
        cutoff_ts = max_timestamp - self.last_days * DAY_SECONDS

        stat = (
            lf.filter((pl.col("event_type") == "listen")
                     & (pl.col("timestamp") > cutoff_ts))
              .group_by("item_id")
              .agg(pl.col("uid").n_unique().alias("unique_users"))
              .sort("unique_users", descending=True)
              .head(self.max_limit)
              .collect()  # materialize LazyFrame
        )
        self.rec = stat["item_id"].to_list()
        self.weights = stat["unique_users"].to_list()

    def recommend(self, uid):
        return self.rec, self.weights
        

class MostPop_by_likes:
    def __init__(self):
        self.max_limit = 2_000
        self.rec = []
        self.weights = []
        self.last_days = 5

    def fit(self, lf: pl.LazyFrame):

        max_timestamp = lf.select(pl.col("timestamp").max()).collect().to_series()[0]
        cutoff_ts = max_timestamp - self.last_days * DAY_SECONDS

        top_df = (
            lf
            .filter(
                (pl.col("event_type") == "like")
                & (pl.col("timestamp") > cutoff_ts)
            )
            .group_by("item_id")
            .agg(pl.len().alias("counts"))
            .sort("counts", descending=True)
            .head(self.max_limit)
            .collect()
        )

        self.rec = top_df["item_id"].to_list()
        self.weights = top_df["counts"].to_list()

    def recommend(self, uid):
        return self.rec, self.weights


class NewItemsLastNDays:
    def __init__(self, days: int = 5, ):
        """
        days     — за сколько последних дней считать новинки
        
        """
        self.max_limit = 2_000
        self.last_days = 5
        self.rec = []
        self.weights = []

    def fit(self, lf: pl.LazyFrame):

        max_timestamp = lf.select(pl.col("timestamp").max()).collect().to_series()[0]
        cutoff_ts = max_timestamp - self.last_days * DAY_SECONDS
        
        history = (
                lf
                .filter(pl.col("timestamp") < cutoff_ts)
                .select(pl.col("item_id").unique())
                .collect()["item_id"]
                .to_list()
            )
        
        current = (
                lf
                .filter(pl.col("timestamp") >= cutoff_ts)
                .select(pl.col("item_id").unique())
                .collect()["item_id"]
                .to_list()
            )

        new = list(set(current) - set(history))
        
        new_items = lf.filter(pl.col("item_id").is_in(new))

        vc = (
            new_items
            .group_by("item_id")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .collect()
        )
        
        self.rec = vc["item_id"].to_list()
        self.weights = vc["count"].to_list()

    def recommend(self, uid):
        return self.rec, self.weights
        


def create_user_item_matrix(dlf: pl.LazyFrame):
    # сразу выбираем только нужные колонки и кастуем типы
    df = (
        dlf
        .select([
            pl.col("uid").cast(pl.Int32).alias("uid"),
            pl.col("item_id").cast(pl.Int32).alias("item_id"),
            pl.col("conf").cast(pl.Int32).alias("conf"),
        ])
        .collect()  # вот здесь LazyFrame -> DataFrame
    )

    rows = df["uid"].to_numpy()
    cols = df["item_id"].to_numpy()
    data = df["conf"].to_numpy()

    mat = coo_matrix((data, (rows, cols))).tocsr()
    return mat


class ALS:
    def __init__(self):
        self.model = AlternatingLeastSquares(factors=128, #Размерность скрытых признаков (эмбеддингов)
                                regularization=0.001,
                                iterations=15,
                                # alpha=1,
                                # use_cg=True,
                                random_state=42,
                                calculate_training_loss = True) 
        self.N = 2000
        
    def fit(self, lf):
        self.user_map, self.item_map = build_id_maps(lf)
        self.reverse_item_map = {v: k for k, v in self.item_map.items()}
        
        lf = map_with_id_maps(lf, self.user_map, self.item_map)
        
        self.matrix = create_user_item_matrix(lf)
        self.model.fit(self.matrix)

    
    def recommend(self, uid):
        uid = int(uid)  #Обязательно принимает int, не конвертирует float
        uid = self.user_map.get(uid, None)
        if uid is None:
            return [], []
            
        # print(self.user_map)
        row = self.matrix[uid]
        if row.nnz == 0:      # nnz = number of non-zero elements
            return [], []
            
        rec_items, w_rec = self.model.recommend(
                                userid=uid,
                                user_items=self.matrix[uid],
                                N=self.N,
                                filter_already_liked_items=False,   
                            )

        orig_ids = [self.reverse_item_map[v] for v in rec_items]

        return list(orig_ids), list(w_rec)



import numpy as np
import scipy.sparse as sp
import polars as pl
from implicit.nearest_neighbours import BM25Recommender
from sklearn.utils.sparsefuncs_fast import inplace_csr_row_normalize_l2


class BM25Rec:
    def __init__(self):
        self.model = BM25Recommender(K=200, K1=0.1, B=0.75, num_threads=0)
        self.N = 2000
        
    def fit(self, lf):
        
        self.user_map, self.item_map = build_id_maps(lf)
        self.reverse_item_map = {v: k for k, v in self.item_map.items()}
        
        lf = map_with_id_maps(lf, self.user_map, self.item_map)

        self.matrix = create_user_item_matrix(lf)

        
        self.matrix = self.matrix.tocsr().astype(np.double) # Алгоритм просит именно double
        inplace_csr_row_normalize_l2(self.matrix)  # X изменится на месте
        
        self.model.fit(self.matrix)

    
    def recommend(self, uid):
        uid = int(uid)  #Обязательно принимает int, не конвертирует float
        uid = self.user_map.get(uid, None)
        if uid is None:
            return [], []
            
        # print(self.user_map)
        row = self.matrix[uid]
        if row.nnz == 0:      # nnz = number of non-zero elements
            return [], []
            
        rec_items, w_rec = self.model.recommend(
                                userid=uid,
                                user_items=self.matrix[uid],
                                N=self.N,  
                            )

        orig_ids = [self.reverse_item_map[v] for v in rec_items]

        return list(orig_ids), list(w_rec)

        


class BPR:
    def __init__(self):
        self.model = BayesianPersonalizedRanking(
                        factors=128,       # размер латентного пространства
                        learning_rate=0.1,
                        regularization=0.01,
                        iterations=300,
                        random_state=42,
                    )

        self.N = 2000
        
    def fit(self, lf):
        self.user_map, self.item_map = build_id_maps(lf)
        self.reverse_item_map = {v: k for k, v in self.item_map.items()}
        
        lf = map_with_id_maps(lf, self.user_map, self.item_map)
        
        self.matrix = create_user_item_matrix(lf)

        self.model.fit(self.matrix.tocsr())

    def recommend(self, uid):
        uid = int(uid)  #Обязательно принимает int, не конвертирует float 
        uid = self.user_map.get(uid, None)
        if uid is None:
            return [], []
        
        row = self.matrix[uid]
        if row.nnz == 0:      # nnz = number of non-zero elements
            return [], []
        rec_items, w_rec = self.model.recommend(
                                userid=uid,
                                user_items=self.matrix[uid],
                                N=self.N,
                                filter_already_liked_items=False,   
                            )
        
        orig_ids = [self.reverse_item_map[v] for v in rec_items]

        return list(orig_ids), list(w_rec)



def add_exponential_decay(train_df: pl.LazyFrame | pl.DataFrame, tau: float):
    # 0, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.5, 1.0, 2
    # Фильтрация по условиям
    train_df = (
        train_df
        .filter(pl.col("played_ratio_pct") > 50)
        .filter(pl.col("event_type") == "listen")
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
    )

    return train_df
    

class ItemKNN:
    def __init__(self):
        self.N = 2000
        
    def fit(self, lf):
        # 0, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.5, 1.0, 2
        hour = 2
        decay = 0.9
        tau = 0.0 if hour == 0 else decay ** (1 / 24 / 60 / 60 / (hour / 24))
        
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
        uid = int(uid) 
        uid = self.user_map.get(uid, None)
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
    

        
class LastListenRecommender:
    def __init__(self):
        self.user_fav_songs = None

    def fit(self, df_merged: pl.DataFrame | pl.LazyFrame):
        self.user_fav_songs = df_merged.collect()
        
    def recommend(self, uid: int):
        if self.user_fav_songs is None:
            raise RuntimeError("Call fit() before recommend().")

        df = (
            self.user_fav_songs
            .filter(pl.col("uid") == uid)
            .sort(
                by=["conf"],
                descending=[True],
            )
        )

        rec = df.get_column("item_id").to_list()
        weights = df.get_column("conf").to_list()  

        return rec, weights



import numpy as np
import polars as pl
from scipy import sparse

class RandomWalkWithRestart:
    def __init__(self):
        pass

    def fit(self, lf_user_track):
        
        df = lf_user_track.filter(pl.col("played_ratio_max")>=50)
        df = df.filter(pl.col("listen_count")>5).collect()  
        # df = lf_user_track.collect()
        
        df_ut = (
            df
            .with_columns(
                (
                    (
                        (
                            pl.col("listen_count").cast(pl.Float64)
                            # * (pl.col("played_ratio_max").cast(pl.Float64) / 100.0)
                        ).log1p()
                        # + 20 * pl.col("like_flag") - 10 * pl.col("dislike_flag")
                    )
                    .clip(1e-6, 10.0)
                ).alias("w")
            )

            # .with_columns(
            #     pl.lit(1.0).alias("w")     # ← ВСЕГДА вес = 1
            # )

                
            .select(
                "uid",
                "item_id",
                "w", #( pl.col("listen_count")).alias("w")
            )
            .group_by(["uid", "item_id"])
            .agg(pl.col("w").sum().alias("weight"))
        )

        # --- track–artist, вес = 1 ---
        df_ta = (
            df
            .select("item_id", "artist_id")
            .unique()
            .with_columns(pl.lit(10.0).alias("weight"))
        )
        
        # --- track–album, вес = 1 ---
        df_tal = (
            df
            .select("item_id", "album_id")
            .unique()
            .with_columns(pl.lit(10.0).alias("weight"))
        )

        # --------------------------------------------------------
        # 2. Кодирование узлов в int и построение CSR-матрицы
        # --------------------------------------------------------
        
        # Уникальные ID по типам узлов
        user_ids = df["uid"].unique().to_list()
        track_ids = df["item_id"].unique().to_list()
        artist_ids = df["artist_id"].unique().to_list()
        album_ids = df["album_id"].unique().to_list()
        
        n_users = len(user_ids)
        n_tracks = len(track_ids)
        n_artists = len(artist_ids)
        n_albums = len(album_ids)
        
        # Смещения для каждого типа узла
        offset_user = 0
        offset_track = offset_user + n_users
        offset_artist = offset_track + n_tracks
        offset_album = offset_artist + n_artists
        
        n_nodes = n_users + n_tracks + n_artists + n_albums
        
        # Маппинги ID → индекс узла
        user2idx = {u: offset_user + i for i, u in enumerate(user_ids)}
        track2idx = {t: offset_track + i for i, t in enumerate(track_ids)}
        artist2idx = {a: offset_artist + i for i, a in enumerate(artist_ids)}
        album2idx = {al: offset_album + i for i, al in enumerate(album_ids)}
        
        # --- user–track рёбра ---
        u_arr = df_ut["uid"].to_list()
        t_arr_ut = df_ut["item_id"].to_list()
        w_ut = df_ut["weight"].to_numpy()
        
        rows_ut = np.fromiter((user2idx[u] for u in u_arr), dtype=np.int64, count=len(u_arr))
        cols_ut = np.fromiter((track2idx[t] for t in t_arr_ut), dtype=np.int64, count=len(t_arr_ut))
        
        # --- track–artist рёбра ---
        t_arr_ta = df_ta["item_id"].to_list()
        a_arr = df_ta["artist_id"].to_list()
        w_ta = df_ta["weight"].to_numpy()
        
        rows_ta = np.fromiter((track2idx[t] for t in t_arr_ta), dtype=np.int64, count=len(t_arr_ta))
        cols_ta = np.fromiter((artist2idx[a] for a in a_arr), dtype=np.int64, count=len(a_arr))
        
        # --- track–album рёбра ---
        t_arr_tal = df_tal["item_id"].to_list()
        al_arr = df_tal["album_id"].to_list()
        w_tal = df_tal["weight"].to_numpy()
        
        rows_tal = np.fromiter((track2idx[t] for t in t_arr_tal), dtype=np.int64, count=len(t_arr_tal))
        cols_tal = np.fromiter((album2idx[al] for al in al_arr), dtype=np.int64, count=len(al_arr))
        
        # Собираем все ориентированные рёбра
        rows_dir = np.concatenate([rows_ut, rows_ta, rows_tal])
        cols_dir = np.concatenate([cols_ut, cols_ta, cols_tal])
        data_dir = np.concatenate([w_ut, w_ta, w_tal]).astype(np.float32)
        
        # Делаем граф неориентированным: дублируем рёбра в обе стороны
        rows_full = np.concatenate([rows_dir, cols_dir])
        cols_full = np.concatenate([cols_dir, rows_dir])
        data_full = np.concatenate([data_dir, data_dir])
        
        # Разреженная матрица смежности A (CSR)
        A = sparse.csr_matrix(
            (data_full, (rows_full, cols_full)),
            shape=(n_nodes, n_nodes),
            dtype=np.float32,
        )

        # --------------------------------------------------------
        # 3. Матрица переходов P и вспомогательные структуры
        # --------------------------------------------------------
        
        # Нормируем строки A → стохастическая матрица переходов P
        row_sums = np.array(A.sum(axis=1)).flatten().astype(np.float64)
        row_sums[row_sums == 0.0] = 1.0  # чтобы избежать деления на 0
        
        D_inv = sparse.diags(1.0 / row_sums)
        P = D_inv @ A  # P: каждая строка — распределение перехода
        
        # Для PPR удобнее использовать P^T
        P_T = P.T.tocsr()
        
        # Для рекомендаций пригодятся:
        # - исходный df (чтобы исключать уже слушанные треки)
        self.df_user_track = df  # просто переименуем для совместимости с твоим кодом


        self.user2idx = user2idx
        self.offset_track = offset_track
        self.n_tracks = n_tracks
        self.P_T = P_T
        self.track_ids = track_ids

    # --------------------------------------------------------
    # 4. Функция Personalized PageRank
    # --------------------------------------------------------
    
    def personalized_pagerank(self, start_idx, alpha=0.85, tol=1e-2, max_iter=2):
        """
        PageRank с персонализацией на одном узле (start_idx).
        Возвращает вектор размерности n_nodes.
        """
        n = self.P_T.shape[0]
    
        # personalization-вектор v
        v = np.zeros(n, dtype=np.float32)
        v[start_idx] = 1.0
    
        # инициализация x
        x = np.full(n, 1.0 / n, dtype=np.float32)
    
        for _ in range(max_iter):
            x_new = alpha * (self.P_T @ x) + (1 - alpha) * v
            # L1-норма разности
            if np.linalg.norm((x_new - x).astype(np.float64), 1) < tol:
                x = x_new
                break
            x = x_new
    
        return x



    def recommend(self, uid, alpha=0.5):

        if uid not in self.user2idx:
            return [], []
    
        start_node_idx = self.user2idx[uid]
    
        # Считаем personalized PageRank
        scores = self.personalized_pagerank(start_node_idx, alpha=alpha)
    
        # Берём только треки (их индексы идут подряд от offset_track)
        track_scores = scores[self.offset_track : self.offset_track + self.n_tracks]
    
        top_idx = np.argsort(-track_scores)
    
        items = [self.track_ids[i] for i in top_idx]
        scores = track_scores[top_idx].tolist()
        return items, scores



##################################################################


class CBF_by_metadata:
    def __init__(self):
        pass

    def fit(self, items_meta, data):
        self.data = data
        items_meta['artist_id'] = items_meta['artist_id'].map(items_meta['artist_id'].value_counts())
        items_meta['album_id'] = items_meta['album_id'].map(items_meta['album_id'].value_counts())
        items_meta['track_length_seconds'] = items_meta['track_length_seconds'].map(items_meta['track_length_seconds'].value_counts())
        
        X = items_meta[["track_length_seconds", "artist_id", "album_id"]]
        self.item_ids = items_meta["item_id"].to_numpy()
        

        X_batch = X.to_numpy().astype("float32")
        # X_batch = np.asarray(X_batch, dtype='float32')  # ensure numpy + float32
        self.X_batch = np.ascontiguousarray(X_batch)    # ensure C-contiguous
        print(self.X_batch)
        
        N, D = self.X_batch.shape
        self.index = faiss.IndexFlatIP(D)
        
        faiss.normalize_L2(self.X_batch)
        self.index.add(self.X_batch)
        
        print("Всего в индексе:", self.index.ntotal)
        
        self.id2pos = {iid: i for i, iid in enumerate(self.item_ids)} # Индексы реальные и в матрице - разные. Нужно для сопоставления

    # Профиль пользователя = средний вектор всех айтемов, которые он слушал.
    def build_user_profile_embed(self, uid):
        listened_items = self.data[self.data["uid"] == uid]["item_id"].values
    
        indices = [self.id2pos[iid] for iid in listened_items if iid in self.id2pos]
        if indices == []:
            return None
        
        # берём вектора всех прослушанных треков
        vectors = self.X_batch[indices]
        
        # пользовательский профиль = средний вектор
        # user_vec = vectors.ьу(axis=0)
        user_vec = np.median(vectors, axis=0)
        user_vec = np.asarray(user_vec)        # → ndarray (1, D)
    
        return np.array([user_vec])

    def similar_tracks(self, vec, k=10): #

        sims, ids = self.index.search(vec, k+1)
    
        sims = sims[0]
        ids = ids[0]
        
        # убираем элемент, где id == pos
        result = []
        for track_pos, score in zip(ids, sims):
            result.append((self.item_ids[track_pos], float(score)))
            if len(result) == k:
                break
    
        res = [pair[0] for pair in result[:k]]
        return res

    def recommend(self, uid, k = 10):
        vec = self.build_user_profile_embed(uid)
        if vec is None:
            return [], []
        # print(vec)
        rec = self.similar_tracks(vec)
        # print(rec)

        return rec, []



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



class CBF_by_embeding:
    def __init__(self):
        pass

    def fit(self, data, index, item_ids):
        self.data = data

        self.index, self.item_ids = index, item_ids
        self.id2pos = {iid: i for i, iid in enumerate(self.item_ids)} # Индексы реальные и в матрице - разные. Нужно для сопоставления

        
        self.total = []
        self.N = 100
        
    # Профиль пользователя = средний вектор всех айтемов, которые он слушал.
    def build_user_profile_embed(self, uid):
        
        listened_items = self.data[self.data["uid"] == uid]["item_id"].values
    
        indices = [self.id2pos[iid] for iid in listened_items if iid in self.id2pos]
        self.total.append(len(indices))
        
        if len(indices) <1:
            return None
        

        vectors = np.vstack([self.index.reconstruct(int(i)) for i in indices]).astype("float32")

        user_vec = np.median(vectors, axis=0)
        user_vec = np.asarray(user_vec)        
    
        return np.array([user_vec])

    def similar_tracks(self, vec): #

        sims, ids = self.index.search(vec, self.N+1)
    
        sims = sims[0]
        ids = ids[0]
        

        result = []
        for track_pos, score in zip(ids, sims):
            result.append((self.item_ids[track_pos], float(score)))

        return [pair[0] for pair in result], [pair[1] for pair in result] 

    def recommend(self, uid):
        vec = self.build_user_profile_embed(uid)
        if vec is None:
            return [], []    
        return self.similar_tracks(vec)



