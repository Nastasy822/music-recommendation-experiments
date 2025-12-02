
from helpers.big_data_helper import *
from models.utils import *

import polars as pl
import faiss
from tqdm import tqdm
import numpy as np
import json


def create_index(filename):
    lf = pl.scan_parquet(filename)
    CHUNK_SIZE = 50_000
    
    first_batch = (
        lf
        .slice(0, CHUNK_SIZE)
        .select(["item_id", "normalized_embed"])
        .collect()
        .to_pandas()
    )

    first_batch = first_batch.dropna()    

    
    emb0 = np.vstack(first_batch["normalized_embed"].to_list()).astype("float32")
    N0, D = emb0.shape
    
    index = faiss.IndexFlatIP(D)   
    index.add(emb0)
    
    all_item_ids = []
    all_item_ids.extend(first_batch["item_id"].tolist())
    
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