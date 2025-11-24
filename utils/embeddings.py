rom utils.data_cleaning import *
from utils.data_preprocess import *
from utils.evaluate import evaluate_model
from utils.cf_utils import *
from utils.models import *
from utils.big_data_helper import *
import polars as pl

from utils.maps_creater import *


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