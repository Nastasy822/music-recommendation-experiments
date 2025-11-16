from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender
from implicit.bpr import BayesianPersonalizedRanking
import numpy as np
import pandas as pd

HOUR_SECONDS = 60 * 60
DAY_SECONDS = 24 * HOUR_SECONDS

class MostPop_by_likes:
    def __init__(self):
        pass

    def fit(self, df):
        stat = df[df["event_type"] == "like"]["item_id"].value_counts()
        self.rec = stat.index.tolist()
        self.weights = stat.tolist()
        

    def recommend(self, uid, k = 10):
        return self.rec[:k], self.weights[:k]

        
class MostPop_by_listen:
    def __init__(self):
        pass

    def fit(self, df):
        stat = (
                 df[df["event_type"] == "listen"]
                .groupby("item_id")["uid"]
                .nunique()
                .sort_values(ascending=False)
                )

        self.rec = stat.index.tolist()
        self.weights = stat.tolist()
        

    def recommend(self, uid, k = 10):
        return self.rec[:k], self.weights[:k]


class ALS:
    def __init__(self):
        self.model = AlternatingLeastSquares(factors=128, #Размерность скрытых признаков (эмбеддингов)
                                regularization=0.05,
                                iterations=30,
                                alpha=1,
                                random_state=42)

    def fit(self, matrix):
        self.matrix = matrix
        self.model.fit(self.matrix)

    def recommend(self, uid, k = 10):
        uid = int(uid)  #Обязательно принимает int, не конвертирует float 
        
        row = self.matrix[uid]
        if row.nnz == 0:      # nnz = number of non-zero elements
            return [], []
        rec_items, w_rec = self.model.recommend(
                                userid=uid,
                                user_items=self.matrix[uid],
                                N=k,
                                filter_already_liked_items=False,   
                            )
        return list(rec_items), list(w_rec)


class ItemItemRec:
    def __init__(self):
        self.model = ItemItemRecommender(K=200)

    def fit(self, matrix):
        self.matrix = matrix.tocsr().astype(np.double) # Алгоритм просит именно double
        self.model.fit(self.matrix)

    def recommend(self, uid, k = 10):
        uid = int(uid)  #Обязательно принимает int, не конвертирует float 
        
        row = self.matrix[uid]
        if row.nnz == 0:      # nnz = number of non-zero elements
            return [], []
        rec_items, w_rec = self.model.recommend(
                                            uid,
                                            self.matrix[uid],
                                            N=k, 
                                        )
        return list(rec_items), list(w_rec)


class BPR:
    def __init__(self):
        self.model = BayesianPersonalizedRanking(
                            factors=128,       # размер латентного пространства
                            learning_rate=0.1,
                            regularization=0.01,
                            iterations=150,
                            random_state=42,
                        )
    
    def fit(self, matrix):
        self.matrix = matrix.tocsr()
        self.model.fit(self.matrix)

    def recommend(self, uid, k = 10):
        uid = int(uid)  #Обязательно принимает int, не конвертирует float 
        
        row = self.matrix[uid]
        if row.nnz == 0:      # nnz = number of non-zero elements
            return [], []
        rec_items, w_rec = self.model.recommend(
                                userid=uid,
                                user_items=self.matrix[uid],
                                N=k,
                                filter_already_liked_items=False,   
                            )
        return list(rec_items), list(w_rec)


class RecentAactivityBasedRecommendation:
    def __init__(self):
        pass

    def fit(self, df_merged):

        likes = df_merged[df_merged["event_type"] == "like"].copy()

        listen = (
                df_merged[df_merged["event_type"] == "listen"]
                .groupby(['uid', 'item_id'])
                .agg({
                    'timestamp': 'count',
                    'artist_id': 'first',
                    'album_id': 'first',
                    
                })
                .reset_index()
            )
        
        listen = listen[listen["timestamp"]>5]

        self.only_likes = likes
        
        likes = pd.concat([likes, listen])

        self.likes = likes

        
    def ger_rec(self, uid, flag= False):
        N_ARTISTS = 100   # сколько самых популярных артистов
        N_TRACKS = 100    # сколько треков на каждого артиста

        user_likes =  self.likes[self.likes["uid"] == uid]
        user_item = list(user_likes["item_id"].unique())
    
        # 2. Считаем, сколько лайков у каждого артиста
        artist_like_counts = (
            user_likes.groupby("artist_id")["item_id"]
            .count()
            .sort_values(ascending=False)
            .reset_index(name="artist_likes")
        )
        
        # 3. Берём топ-10 артистов
        top_artists = artist_like_counts.head(N_ARTISTS).copy()
        top_artists["artist_rank"] = range(1, len(top_artists) + 1)
        
        # 4. Оставляем лайки только по этим артистам ГЛОБАЛЬНО для артистою ЮЗЕРА
        if flag:
            likes_top = self.likes[self.likes["artist_id"].isin(top_artists["artist_id"])]
        else:
            likes_top = user_likes[user_likes["artist_id"].isin(top_artists["artist_id"])]
        
    
        # 5. Считаем лайки по (artist_id, item_id) — популярность треков у артиста
        track_counts = (
            likes_top.groupby(["artist_id", "item_id"])
            .size()
            .reset_index(name="track_likes")
        )
        
        # 6. Подтягиваем к трекам общие лайки артиста и его ранг
        track_counts = track_counts.merge(top_artists, on="artist_id", how="left")
        
        # 7. Сортируем: сначала по популярности артиста, потом по популярности трека
        track_counts = track_counts.sort_values(
            ["artist_rank", "track_likes"],
            ascending=[True, False]
        )
        
        # 8. Берём топ N_TRACKS треков для каждого артиста и добавляем ранг трека
        top_tracks_per_artist = (
            track_counts
            .groupby("artist_id")
            .head(N_TRACKS)
            .copy()
        )
        
        top_tracks_per_artist["track_rank"] = (
            top_tracks_per_artist.groupby("artist_id")["track_likes"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
        
        # 9. Финальная сортировка для красивого вывода
        top_tracks_per_artist = top_tracks_per_artist.sort_values(
            ["artist_rank", "track_rank"]
        ).reset_index(drop=True)
        # print(top_tracks_per_artist)
        
        if flag:
            top_tracks_per_artist = top_tracks_per_artist[~top_tracks_per_artist["item_id"].isin(user_item)]


        top_tracks_per_artist = top_tracks_per_artist.sort_values("track_likes", ascending = False)

        return top_tracks_per_artist[:10]["item_id"].tolist(), top_tracks_per_artist[:10]["artist_rank"].tolist()

    def recommend(self, uid, k = 10):
        rec, weights = self.ger_rec(uid) 
        return rec[:k], weights[:k]


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


import pandas as pd
import polars as pl


import faiss
from tqdm import tqdm
from sklearn.pipeline import Pipeline


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
        
    # Профиль пользователя = средний вектор всех айтемов, которые он слушал.
    def build_user_profile_embed(self, uid):
        
        listened_items = self.data[self.data["uid"] == uid]["item_id"].values
    
        indices = [self.id2pos[iid] for iid in listened_items if iid in self.id2pos]
        self.total.append(len(indices))
        
        if len(indices) <5:
            return None
        

        vectors = np.vstack([self.index.reconstruct(int(i)) for i in indices]).astype("float32")

        user_vec = np.median(vectors, axis=0)
        user_vec = np.asarray(user_vec)        
    
        return np.array([user_vec])

    def similar_tracks(self, vec, k=10): #

        sims, ids = self.index.search(vec, k+1)
    
        sims = sims[0]
        ids = ids[0]
        

        result = []
        for track_pos, score in zip(ids, sims):
            result.append((self.item_ids[track_pos], float(score)))
            if len(result) == k:
                break

        return [pair[0] for pair in result[:k]], [pair[1] for pair in result[:k]] 

    def recommend(self, uid, k = 10):
        vec = self.build_user_profile_embed(uid)
        if vec is None:
            return [], []    
        return self.similar_tracks(vec)


