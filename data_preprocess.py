def codic_of_index(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    user_id_map = {uid: idx for idx, uid in enumerate(train_df["uid"].unique())}
    item_id_map = {iid: idx for idx, iid in enumerate(train_df["item_id"].unique())}
    
    train_df["uid"] = train_df["uid"].map(user_id_map)
    train_df["item_id"] = train_df["item_id"].map(item_id_map)
    
    test_df["uid"] = test_df["uid"].map(user_id_map)
    test_df["item_id"] = test_df["item_id"].map(item_id_map)

    return train_df, test_df, item_id_map


def train_test_split(data, test_size, train_size, gap_size = 30):
    
    
    
    HOUR_SECONDS = 60 * 60
    DAY_SECONDS = 24 * HOUR_SECONDS
    
    GAP_SIZE = HOUR_SECONDS // 2
    TEST_SIZE = test_size * DAY_SECONDS

    overall_days = (data["timestamp"].max() - data["timestamp"].min())//DAY_SECONDS
    
    data = data[data['timestamp'] > (overall_days-train_size) * DAY_SECONDS]
    
    LAST_TIMESTAMP = data["timestamp"].max()
    
    TEST_TIMESTAMP = LAST_TIMESTAMP - TEST_SIZE
    
    TRAIN_TIMESTAMP = TEST_TIMESTAMP - GAP_SIZE
    
    train_df = data[data['timestamp'] < TRAIN_TIMESTAMP]
    test_df = data[data['timestamp'] > TEST_TIMESTAMP]

    return train_df, test_df
    
    
    