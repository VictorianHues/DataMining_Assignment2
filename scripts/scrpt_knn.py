import pandas as pd
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ndcg_score

# ----------- 1. Load and Sample the Data -----------

def load_sample_data(file_path, n_searches=10000, random_state=42):
    df = pd.read_csv(file_path)
    if n_searches is not None:
        sample_ids = df['srch_id'].drop_duplicates().sample(n=n_searches, random_state=random_state)
        sampled_df = df[df['srch_id'].isin(sample_ids)].copy()
    else:
        sampled_df = df.copy()
    return sampled_df.reset_index(drop=True)

# ----------- 2. Prepare Features and Labels -----------

def prepare_features_labels(df):
    df['relevance'] = 5 * df['booking_bool'] + df['click_bool']
    
    features = [
        'prop_starrating', 'prop_review_score', 'prop_brand_bool',
        'prop_location_score1', 'price_usd', 'promotion_flag',
        'srch_length_of_stay', 'srch_booking_window', 'price_z'
    ]
    
    X = df[features].fillna(-1)
    y = df['relevance']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, features

# ----------- 3. Split by Search ID -----------

def search_based_split(df, X, y, test_size=0.2, random_state=42):
    srch_ids = df['srch_id'].unique()
    train_ids, test_ids = train_test_split(srch_ids, test_size=test_size, random_state=random_state)
    
    train_idx = df['srch_id'].isin(train_ids)
    
    return (
        X[train_idx], y[train_idx],
        X[~train_idx], y[~train_idx],
        df[~train_idx].copy()
    )

# ----------- 4. Train and Predict -----------

def train_knn_predict(X_train, y_train, X_test, k=5):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn.predict_proba(X_test)[:, 1]  # Probability of relevance

# ----------- 5. Evaluate NDCG -----------

def ndcg_per_search(test_df, k=5):
    scores = []
    for srch_id, group in test_df.groupby('srch_id'):
        if group['relevance'].sum() == 0:
            continue
        true = group['relevance'].values.reshape(1, -1)
        pred = group['predicted_score'].values.reshape(1, -1)
        scores.append(ndcg_score(true, pred, k=k))
    return np.mean(scores)

# ----------- 6. Main Pipeline -----------

def main():
    import os
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "..", 'data', 'training_set_VU_DM_engineered.csv')

    print("Loading and sampling data...")
    df = load_sample_data(file_path)

    print("Preparing features...")
    X, y, features = prepare_features_labels(df)

    print("Splitting data...")
    X_train, y_train, X_test, y_test, test_df = search_based_split(df, X, y)

    print("Training KNN...")
    test_df['predicted_score'] = train_knn_predict(X_train, y_train, X_test)

    print("Ranking within each search (test)...")
    test_df['rank_within_search'] = test_df.groupby('srch_id')['predicted_score'] \
                                           .rank(method='first', ascending=False)

    print("Evaluating NDCG@5 on test data...")
    ndcg_test = ndcg_per_search(test_df, k=5)

    print("Predicting on training data...")
    train_ids = df['srch_id'].isin(df['srch_id'].unique()) & ~df['srch_id'].isin(test_df['srch_id'])
    train_df = df[train_ids].copy()
    train_df['relevance'] = 5 * train_df['booking_bool'] + train_df['click_bool']
    train_X = train_df[features].fillna(-1)
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train_X_scaled = scaler.fit(X).transform(train_X)  # Same scaling logic
    train_df['predicted_score'] = train_knn_predict(X_train, y_train, train_X_scaled)

    train_df['rank_within_search'] = train_df.groupby('srch_id')['predicted_score'] \
                                             .rank(method='first', ascending=False)

    print("Evaluating NDCG@5 on train data...")
    ndcg_train = ndcg_per_search(train_df, k=5)

    print(f"\nTrain NDCG@5: {ndcg_train:.5f}")
    print(f"Test  NDCG@5: {ndcg_test:.5f}")


if __name__ == "__main__":
    main()
