import numpy as np
import polars as pl
from scipy import sparse
from models.utils import merge_data_by_count
from models.base_model import BaseModel
import json

class RandomWalkWithRestart(BaseModel):
    def __init__(self):
        super().__init__()

        self.played_ratio_max = self.params.RandomWalkWithRestart.played_ratio_max
        self.low_weights_limit = self.params.RandomWalkWithRestart.low_weights_limit
        self.high_weights_limit = self.params.RandomWalkWithRestart.high_weights_limit
        self.artist_edges_weights = self.params.RandomWalkWithRestart.artist_edges_weights
        self.album_edges_weights = self.params.RandomWalkWithRestart.album_edges_weights
        self.listen_count_limit = self.params.RandomWalkWithRestart.listen_count_limit
        self.alpha = self.params.RandomWalkWithRestart.alpha
        self.tol = self.params.RandomWalkWithRestart.tol
        self.max_iter = self.params.RandomWalkWithRestart.max_iter


    def fit(self, train_df):
        
        with open("data/item_map.json", "r", encoding="utf-8") as f:
            item_map = json.load(f)

        item_map = {int(k): v for k, v in item_map.items()}

        items_meta = (
            pl.scan_parquet("data/source/items_meta.parquet")
            .with_columns(
                pl.col("item_id").replace(item_map)
            )
            .unique(subset=["item_id"])
            .drop_nulls()
        )

        df_merged = merge_data_by_count(train_df)
        lf_user_track = df_merged.join(
                items_meta.select([self.item_id_column, self.artist_id_column, self.album_id_column]),
                on=self.item_id_column,
                how="left",
            )

        
        self.df_user_track = lf_user_track.filter(pl.col(self.played_ratio_max_column)>=self.played_ratio_max)
        self.df_user_track  = self.df_user_track .filter(pl.col(self.listen_count_column)>self.listen_count_limit).collect()  
        
        df_ut = (
            self.df_user_track 
            .with_columns(
                (
                    pl.col(self.listen_count_column)
                    .cast(pl.Float64)
                    .log1p()
                    .clip(self.low_weights_limit , self.high_weights_limit)
                )
            .alias(self.weights_column)
            )
                
            .select(
                self.user_id_column,
                self.item_id_column,
                self.weights_column,
            )
            .group_by([self.user_id_column, self.item_id_column])
            .agg(pl.col(self.weights_column).sum().alias(self.weights_column))
        )

        # --- track–artist
        df_ta = (
            self.df_user_track 
            .select(self.item_id_column, self.artist_id_column)
            .unique()
            .with_columns(pl.lit(self.artist_edges_weights).alias(self.weights_column))
        )
        
        # --- track–album
        df_tal = (
            self.df_user_track 
            .select(self.item_id_column, self.album_id_column)
            .unique()
            .with_columns(pl.lit(self.album_edges_weights).alias(self.weights_column))
        )

        
        # unique ID by nodes
        user_ids = self.df_user_track[self.user_id_column].unique().to_list()
        self.track_ids = self.df_user_track[self.item_id_column].unique().to_list()
        artist_ids = self.df_user_track[self.artist_id_column].unique().to_list()
        album_ids = self.df_user_track[self.album_id_column].unique().to_list()
        
        n_users = len(user_ids)
        self.n_tracks = len(self.track_ids)
        n_artists = len(artist_ids)
        n_albums = len(album_ids)
        
        # bias for nodes 
        offset_user = 0
        self.offset_track = offset_user + n_users
        offset_artist = self.offset_track + self.n_tracks
        offset_album = offset_artist + n_artists
        
        n_nodes = n_users + self.n_tracks + n_artists + n_albums
        
        # mapping ID → index node 
        self.user2idx = {u: offset_user + i for i, u in enumerate(user_ids)}
        track2idx = {t: self.offset_track + i for i, t in enumerate(self.track_ids)}
        artist2idx = {a: offset_artist + i for i, a in enumerate(artist_ids)}
        album2idx = {al: offset_album + i for i, al in enumerate(album_ids)}
        
        # --- user–track edges ---
        u_arr = df_ut[self.user_id_column].to_list()
        t_arr_ut = df_ut[self.item_id_column].to_list()
        w_ut = df_ut[self.weights_column].to_numpy()
        
        rows_ut = np.fromiter((self.user2idx[u] for u in u_arr), dtype=np.int64, count=len(u_arr))
        cols_ut = np.fromiter((track2idx[t] for t in t_arr_ut), dtype=np.int64, count=len(t_arr_ut))
        
        # --- track–artist edges ---
        t_arr_ta = df_ta[self.item_id_column].to_list()
        a_arr = df_ta[self.artist_id_column].to_list()
        w_ta = df_ta[self.weights_column].to_numpy()
        
        rows_ta = np.fromiter((track2idx[t] for t in t_arr_ta), dtype=np.int64, count=len(t_arr_ta))
        cols_ta = np.fromiter((artist2idx[a] for a in a_arr), dtype=np.int64, count=len(a_arr))
        
        # --- track–album edges ---
        t_arr_tal = df_tal[self.item_id_column].to_list()
        al_arr = df_tal[self.album_id_column].to_list()
        w_tal = df_tal[self.weights_column].to_numpy()
        
        rows_tal = np.fromiter((track2idx[t] for t in t_arr_tal), dtype=np.int64, count=len(t_arr_tal))
        cols_tal = np.fromiter((album2idx[al] for al in al_arr), dtype=np.int64, count=len(al_arr))
        
        # collect all edges
        rows_dir = np.concatenate([rows_ut, rows_ta, rows_tal])
        cols_dir = np.concatenate([cols_ut, cols_ta, cols_tal])
        data_dir = np.concatenate([w_ut, w_ta, w_tal]).astype(np.float32)
        
        # Making the graph undirected: duplicating edges in both directions
        rows_full = np.concatenate([rows_dir, cols_dir])
        cols_full = np.concatenate([cols_dir, rows_dir])
        data_full = np.concatenate([data_dir, data_dir])
        
        # Sparse adjacency matrix A (CSR)
        A = sparse.csr_matrix(
            (data_full, (rows_full, cols_full)),
            shape=(n_nodes, n_nodes),
            dtype=np.float32,
        )
        
        # We normalize the rows of A → stochastic transition matrix P
        row_sums = np.array(A.sum(axis=1)).flatten().astype(np.float64)
        row_sums[row_sums == 0.0] = 1.0  # to avoid division by 0
        
        D_inv = sparse.diags(1.0 / row_sums)
        P = D_inv @ A  # P: each line is a transition distribution
        
        # For PPR it is more convenient to use P^T
        self.P_T = P.T.tocsr()
        
    
    def personalized_pagerank(self, start_idx):

        n = self.P_T.shape[0]
    
        v = np.zeros(n, dtype=np.float32)
        v[start_idx] = 1.0

        x = np.full(n, 1.0 / n, dtype=np.float32)
    
        for _ in range(self.max_iter):
            x_new = self.alpha * (self.P_T @ x) + (1 - self.alpha) * v

            if np.linalg.norm((x_new - x).astype(np.float64), 1) < self.tol:
                x = x_new
                break
            x = x_new
    
        return x


    def recommend(self, uid):

        if uid not in self.user2idx:
            return [], []
    
        start_node_idx = self.user2idx[uid]
    
        scores = self.personalized_pagerank(start_node_idx)
    
        #We take only tracks (their indices follow a sequence from offset_track)
        track_scores = scores[self.offset_track : self.offset_track + self.n_tracks]
    
        top_idx = np.argsort(-track_scores)
    
        items = [self.track_ids[i] for i in top_idx]
        scores = track_scores[top_idx].tolist()
        return items, scores