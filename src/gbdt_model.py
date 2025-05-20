import lightgbm as lgb

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import GroupKFold
from sklearn.metrics import ndcg_score

def get_features():
    return [
        'prop_country_id', 'prop_starrating', 'prop_review_score',
        'prop_brand_bool', 'prop_location_score1',
        'price_usd', 'promotion_flag', 'srch_destination_id',
        'srch_length_of_stay',
        'srch_adults_count', 'srch_children_count', 'srch_room_count',
        'sum_comp_rate', 'sum_comp_inv', 'log_price',
        'price_rank', 'review_rank', 'location_score_rank',
        'is_cheapest', 'is_best_rated',
        'is_top3_cheapest', 'price_percentile', 'price_z',
        'prop_booking_rate', 'prop_click_rate',
        'prop_avg_price', 'prop_avg_review_score',
        'dest_avg_price', 'country_avg_star'
    ]

def create_labels(train_df):
    train_df['relevance'] = train_df['booking_bool'] * 2 + train_df['click_bool'] * 1
    train_df['relevance'] = train_df['relevance'].astype(int)
    return train_df

def split_data(train_df, features):
    gkf = GroupKFold(n_splits=5)
    for train_idx, val_idx in gkf.split(train_df[features], train_df['relevance'], groups=train_df['srch_id']):
        break

    X_train = train_df.iloc[train_idx][features]
    y_train = train_df.iloc[train_idx]['relevance']
    srch_train = train_df.iloc[train_idx]['srch_id']

    X_val = train_df.iloc[val_idx][features]
    y_val = train_df.iloc[val_idx]['relevance']
    srch_val = train_df.iloc[val_idx]['srch_id']

    group_train = srch_train.groupby(srch_train).size().to_numpy()
    group_val = srch_val.groupby(srch_val).size().to_numpy()

    return X_train, y_train, group_train, X_val, y_val, group_val, val_idx

def train_gbdt(X_train, y_train, group_train, X_val, y_val, group_val):
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [5],
        'learning_rate': 0.15,
        'num_leaves': 128,
        'min_data_in_leaf': 50,
        'boosting_type': 'dart',
        'drop_rate': 0.1,
        'verbosity': -1
    }

    lgb_train = lgb.Dataset(X_train, label=y_train, group=group_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, group=group_val)

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=200,
        valid_sets=[lgb_val],
        valid_names=['val']
    )
    return model

def evaluate_ndcg(model, train_df, X_val, val_idx):
    val_df = train_df.iloc[val_idx].copy()
    val_df['predicted_score'] = model.predict(X_val)

    def ndcg_per_group(group):
        true = group['relevance'].values.reshape(1, -1)
        pred = group['predicted_score'].values.reshape(1, -1)
        return ndcg_score(true, pred, k=5)

    ndcg_val = val_df.groupby('srch_id').apply(ndcg_per_group).mean()
    print(f"\n Validation NDCG@5: {ndcg_val:.5f}\n")