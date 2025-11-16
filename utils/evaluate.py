from tqdm import tqdm
from utils.metrics import ndcg_at_k, recall_at_k
import numpy as np

def evaluate_model(model, train_df, test_df, k=10):
    test_df = test_df[test_df["event_type"]!="dislike"]
    grouped_users = test_df.groupby("uid")
    overall_recall = []
    overall_ndcg = []
    for uid, group in tqdm(grouped_users):
        # print(list(group["item_id"]))
        user_true = list(set(group["item_id"]))
        
        rec, _ = model.recommend(uid, k)
        recall = recall_at_k(rec, user_true,  10)
        ndcg = ndcg_at_k(rec ,user_true,  10)
        
        overall_recall.append(recall)
        overall_ndcg.append(ndcg)
    
    print("Mean Recall@10:", np.mean(overall_recall))
    print("Mean NDCG@10:",np.mean(overall_ndcg))
