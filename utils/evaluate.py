from tqdm import tqdm
from utils.metrics import ndcg_at_k, recall_at_k
import numpy as np


def get_user_listened_items(history, uid):
    return history[history["uid"] == uid]["item_id"].unique().tolist()


def filtering_listened_items(train_df, uid, rec, weights, k=None):

    user_listened_items = set(get_user_listened_items(train_df, uid))
    
    filtered_rec = []
    filtered_weights = []


    for item, weight in zip(rec, weights):
        if item in user_listened_items:
            continue
        filtered_rec.append(item)
        filtered_weights.append(weight)
        

    return filtered_rec, filtered_weights

    
def evaluate_model(model, train_df, test_df, k=10):
    test_df = test_df[test_df["event_type"]!="dislike"]
    grouped_users = test_df.groupby("uid")
    overall_recall = []
    overall_ndcg = []
    for uid, group in tqdm(grouped_users):
        # print(list(group["item_id"]))
        user_true = list(set(group["item_id"]))

        # print("-------------------")
        rec, weights = model.recommend(uid)
        rec, weights = filtering_listened_items(train_df, uid, rec, weights)
    
        # print(rec)
        user_listened_items = set(get_user_listened_items(train_df, uid))
        perfect = list(set(user_true)-set(user_listened_items))

        if len(perfect) == 0:
            continue   # skip user

        recall = recall_at_k(rec, perfect,  10)
        ndcg = ndcg_at_k(rec ,perfect,  10)

        # print(uid, ndcg, len(rec))

        
        overall_recall.append(recall)
        overall_ndcg.append(ndcg)
    
    print("Mean Recall@10:", np.mean(overall_recall))
    print("Mean NDCG@10:",np.mean(overall_ndcg))