import numpy as np
import polars as pl
from scipy import sparse
from models.utils import merge_data_by_count
from models.base_model import BaseModel


class RandomWalkWithRestart(BaseModel):
    def __init__(self):
        pass

    def fit(self, train_df, items_meta):

        df_merged = merge_data_by_count(train_df)
        lf_user_track = df_merged.join(
                items_meta.select(["item_id", "artist_id", "album_id"]),
                on="item_id",
                how="left",
            )

        
        df = lf_user_track.filter(pl.col("played_ratio_max")>=50)
        df = df.filter(pl.col("listen_count")>5).collect()  
        
        df_ut = (
            df
            .with_columns(
                (
                    (
                        (
                            pl.col("listen_count").cast(pl.Float64)
                        ).log1p()
                    )
                    .clip(1e-6, 10.0)
                ).alias("w")
            )
                
            .select(
                "uid",
                "item_id",
                "w",
            )
            .group_by(["uid", "item_id"])
            .agg(pl.col("w").sum().alias("weight"))
        )

        # --- track–artist, вес = 10 ---
        df_ta = (
            df
            .select("item_id", "artist_id")
            .unique()
            .with_columns(pl.lit(10.0).alias("weight"))
        )
        
        # --- track–album, вес = 10 ---
        df_tal = (
            df
            .select("item_id", "album_id")
            .unique()
            .with_columns(pl.lit(10.0).alias("weight"))
        )

        
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
        
        # Нормируем строки A → стохастическая матрица переходов P
        row_sums = np.array(A.sum(axis=1)).flatten().astype(np.float64)
        row_sums[row_sums == 0.0] = 1.0  # чтобы избежать деления на 0
        
        D_inv = sparse.diags(1.0 / row_sums)
        P = D_inv @ A  # P: каждая строка — распределение перехода
        
        # Для PPR удобнее использовать P^T
        P_T = P.T.tocsr()
        
        self.df_user_track = df 

        self.user2idx = user2idx
        self.offset_track = offset_track
        self.n_tracks = n_tracks
        self.P_T = P_T
        self.track_ids = track_ids

    # --------------------------------------------------------
    # 4. Функция Personalized PageRank
    # --------------------------------------------------------
    
    def personalized_pagerank(self, start_idx, alpha=0.85, tol=1e-2, max_iter=2):

        n = self.P_T.shape[0]
    
        v = np.zeros(n, dtype=np.float32)
        v[start_idx] = 1.0

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