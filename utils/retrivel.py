from catboost import CatBoostRanker
from utils.evaluate import filtering_listened_items


class CandidateGenerator:
    def __init__(self):
        self.trend_model = MostPop_by_likes()
        self.new_items_model = NewItemsLastNDays()
        self.als = ALS()
        self.item_knn = ItemKNN()
        self.graph = RandomWalkWithRestart()
        self.embed = CBF_by_embeding_kmean()
    
        self.limit_trend = 100
        self.limit_new = 100
        self.limit_als = 200
        self.limit_knn = 200
        self.limit_graph = 200
        self.limit_embed = 200
        

    def fit(self, train_df, items_meta):
                      
        print("Train recomended models")
        self.trend_model.fit(train_df)
        self.new_items_model.fit(train_df) 

        self.als.fit(train_df)
        self.item_knn.fit(train_df)
        
        self.graph.fit(train_df, items_meta)

        self.embed.fit(train_df)


    def gather_candidates(self, uid):

        candidates = {}
        
        rec, weights = self.trend_model.recommend(uid)
        rec, weights = rec[:self.limit_trend], weights[:self.limit_trend]
        weights = [i for i in range(len(rec))]
        for item, score in zip(rec, weights):
            candidates.setdefault(item, {})['trend'] = score
    
        rec, weights = self.new_items_model.recommend(uid)
        rec, weights = rec[:self.limit_new], weights[:self.limit_new]

        weights = [i for i in range(len(rec))]
        for item, score in zip(rec, weights):
            candidates.setdefault(item, {})['new'] = score

        rec, weights = self.als.recommend(uid)
        rec, weights = rec[:self.limit_als], weights[:self.limit_als]

        # weights = [i for i in range(len(rec))]
        for item, score in zip(rec, weights):
            candidates.setdefault(item, {})['als'] = score

        rec, weights = self.item_knn.recommend(uid)
        rec, weights = rec[:self.limit_knn], weights[:self.limit_knn]
    
        for item, score in zip(rec, weights):
            candidates.setdefault(item, {})['item_knn'] = score


        rec, weights = self.graph.recommend(uid)
        rec, weights = rec[:self.limit_graph], weights[:self.limit_graph]
    
        for item, score in zip(rec, weights):
            candidates.setdefault(item, {})['graph'] = score


        rec, weights = self.embed.recommend(uid)
        rec, weights = rec[:self.limit_embed], weights[:self.limit_embed]
    
        for item, score in zip(rec, weights):
            candidates.setdefault(item, {})['embed'] = score
        
        return candidates

    def make_features(self, user_id, candidates):
        rows = []

        for item_id, scores in candidates.items():
            row = {
                # "user_id": user_id,
                "item_id": item_id,
                "score_item_knn":   scores.get("item_knn", 0),
                "score_trend":      scores.get("trend", 0),
                "score_als":   scores.get("als", 0),
                "score_new":   scores.get("new", 0),
                "score_graph":   scores.get("graph", 0),
                "score_embed":   scores.get("embed", 0),
            }
    
            rows.append(row)
        
        return rows