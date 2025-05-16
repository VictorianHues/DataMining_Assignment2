import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

def load_feature_data(path):
    df = pd.read_csv(path)
    srch_ids = df['srch_id'].values
    X = df.drop(columns=['srch_id']).fillna(0).values
    return X, srch_ids

def fit_knn_model(X, k):
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X)
    return knn

def get_k_neighbors(knn, X, srch_ids, target_index, k):
    distances, indices = knn.kneighbors(X)
    neighbor_indices = indices[:, 1:k]  # Exclude self
    target_srch_id = srch_ids[target_index]
    neighbor_srch_ids = srch_ids[neighbor_indices[target_index]]
    return target_srch_id, neighbor_srch_ids

def get_props_seen_by_target(full_data_path, target_srch_id):
    props_seen = set()
    for chunk in pd.read_csv(full_data_path, chunksize=100000):
        filtered = chunk[chunk['srch_id'] == target_srch_id]
        props_seen.update(filtered['prop_id'].unique())
    return props_seen

def get_neighbor_props(full_data_path, neighbor_srch_ids, click_weight, book_weight):
    prop_scores = {}

    for chunk in pd.read_csv(full_data_path, chunksize=100000):
        filtered = chunk[chunk['srch_id'].isin(neighbor_srch_ids)]

        for _, row in filtered.iterrows():
            prop_id = row['prop_id']
            click = row['click_bool']
            book = row['booking_bool']

            if prop_id not in prop_scores:
                prop_scores[prop_id] = {'clicks': 0, 'books': 0, 'total': 0}

            prop_scores[prop_id]['clicks'] += click
            prop_scores[prop_id]['books'] += book
            prop_scores[prop_id]['total'] += 1

    data = []
    for prop_id, vals in prop_scores.items():
        if vals['total'] > 0:
            click_rate = vals['clicks'] / vals['total']
            book_rate = vals['books'] / vals['total']
            combined_score = click_weight * click_rate + book_weight * book_rate
            data.append({'prop_id': prop_id, 'combined_score': combined_score})

    return pd.DataFrame(data)

def recommend_properties(target_props, neighbor_props, top_n):
    unseen_props = neighbor_props[~neighbor_props['prop_id'].isin(target_props)]
    sorted_props = unseen_props.sort_values(by='combined_score', ascending=False)
    return sorted_props.drop_duplicates(subset='prop_id').head(top_n)

def recommend_for_all_users(
    agg_path,
    full_data_path,
    output_path,
    k_neighbors,
    top_n,
    click_weight,
    book_weight
):
    from tqdm import tqdm

    X, srch_ids = load_feature_data(agg_path)
    knn = fit_knn_model(X, k_neighbors)

    output = []

    for idx in tqdm(range(len(srch_ids)), desc="Generating recommendations"):
        try:
            target_srch_id, neighbor_srch_ids = get_k_neighbors(knn, X, srch_ids, idx, k_neighbors)
            props_seen = get_props_seen_by_target(full_data_path, target_srch_id)
            neighbor_props = get_neighbor_props(full_data_path, neighbor_srch_ids, click_weight, book_weight)
            recommendations = recommend_properties(props_seen, neighbor_props, top_n)

            for prop_id in recommendations['prop_id']:
                output.append({'srch_id': target_srch_id, 'prop_id': prop_id})
        except Exception as e:
            print(f"Skipping index {idx} due to error: {e}")

    output_df = pd.DataFrame(output)
    output_df.to_csv(output_path, index=False)
    print(f"Recommendations saved to {output_path}")
