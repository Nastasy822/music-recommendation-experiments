from scipy.sparse import coo_matrix
import numpy as np

def merge_data_by_count(train_df):
    # Делай отрицательное
    train_df_implicit = (
        train_df[train_df["event_type"] == "listen"]
        .groupby(['uid', 'item_id'])
        .agg({
            'timestamp': 'count',
            'played_ratio_pct': 'mean'   # среднее процент прослушивания по всем сессиям
        })
        .reset_index()
    )
    
    train_df_dislike = train_df[train_df["event_type"]=="dislike"].groupby(['uid','item_id'])['timestamp'].count().reset_index()
    train_df_dislike['timestamp'] = - train_df_dislike['timestamp']
    train_df_dislike.loc[train_df_dislike["timestamp"] < 0, "timestamp"] = 1
    
    
    
    train_df_like = train_df[train_df["event_type"]=="like"].groupby(['uid','item_id'])['timestamp'].count().reset_index()
    train_df_like["timestamp"] = 1
    
    
    train_merge = (
        train_df_implicit
        .merge(train_df_dislike, on=['uid', 'item_id'], how='outer')
        .merge(train_df_like, on=['uid', 'item_id'], how='outer')
        .fillna(0)
    )
    train_merge = train_merge.rename(columns={"timestamp": "have_like",
                            "timestamp_y": "have_dislike",
                            "timestamp_x": "listen_count",
                           })
    
    return train_merge

def create_user_item_matrix(df):

    rows = df["uid"].to_numpy(np.int32) 
    cols = df["item_id"].to_numpy(np.int32) 
    data = df["conf"].to_numpy(np.int32) 
    
    user_item_matrix = coo_matrix((data, (rows, cols)))
    
    # сжатие матрицы
    user_item_matrix = user_item_matrix.tocsr()

    return user_item_matrix
