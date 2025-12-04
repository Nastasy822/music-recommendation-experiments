
from helpers.big_data_helper import *
from models.utils import *
from helpers.clusterisations import SphericalMeanShift

import polars as pl
import faiss
from tqdm import tqdm
import numpy as np
import json
from models.base_model import BaseModel
import pickle


class KMeansEmbedding(BaseModel):
    def __init__(self):
        super().__init__()
        self.clusterizer = SphericalMeanShift(  kappa=self.params.KMeansEmbedding.SphericalMeanShift.kappa,          
                                                max_iter=self.params.KMeansEmbedding.SphericalMeanShift.max_iter,
                                                tol=self.params.KMeansEmbedding.SphericalMeanShift.tol,            
                                                merge_angle_deg=self.params.KMeansEmbedding.SphericalMeanShift.merge_angle_deg )          

        self.jitter_level = self.params.KMeansEmbedding.jitter_level
        self.use_jitter = self.params.KMeansEmbedding.use_jitter

        self.embeddings_path = self.params.datasets.filtered_embeddings
        self.embeddings_index_path = self.params.datasets.embeddings_index
        self.index_item_ids_map_path = self.params.datasets.index_item_ids_map

    def fit(self, data):

        self.index = faiss.read_index(self.embeddings_index_path)

        with open(self.index_item_ids_map_path, "rb") as f:
            self.item_ids = pickle.load(f)

        self.id2pos = {iid: i for i, iid in enumerate(self.item_ids)} 
        
        self.history = build_user_listened_items(data)



    def spherical_jitter(self, embeddings, level=0.02):
        noise = np.random.normal(0, 1, embeddings.shape)
        noise = noise - (noise * embeddings).sum(axis=1, keepdims=True) * embeddings
        noise = noise / np.linalg.norm(noise, axis=1, keepdims=True)
        emb_noisy = embeddings + level * noise
        emb_noisy = emb_noisy / np.linalg.norm(emb_noisy, axis=1, keepdims=True)
        return emb_noisy

        
    def build_user_profile_embed(self, uid):

        if uid not in self.history:
            return None
        
        listened_items = self.history[uid]
        indices = [self.id2pos[iid] for iid in listened_items if iid in self.id2pos]
        
        if len(indices)==0:
            return None
                
        vectors = np.vstack([self.index.reconstruct(int(i)) for i in indices]).astype("float32")
        embeddings = np.array(vectors, dtype="float32")
        
        if self.use_jitter: 
            embeddings = self.spherical_jitter(embeddings, self.jitter_level)  
        
        labels, centers =  self.clusterizer.fit_predict(embeddings)

        return centers

    def similar_tracks(self, vecs):

        vecs = np.asarray(vecs, dtype=np.float32)

        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)

        n_queries = len(vecs)
        k = max(1, self.N // n_queries)

        sims, ids = self.index.search(vecs, k)

        flat_ids = ids.ravel()
        flat_sims = sims.ravel()

        order = np.argsort(-flat_sims)

        sorted_ids = flat_ids[order]
        sorted_sims = flat_sims[order]

        all_items = [self.item_ids[i] for i in sorted_ids]
        all_scores = sorted_sims.tolist()

        return all_items, all_scores

    def recommend(self, uid):
        vectors = self.build_user_profile_embed(uid)

        if vectors is None:
            return [], []    
        return self.similar_tracks(vectors)
        