import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def load_data(aggregated_path: str, original_path: str):
    agg_df = pd.read_csv(aggregated_path)
    original_df = pd.read_csv(
        original_path,
        usecols=['srch_id', 'prop_id', 'prop_review_score', 'price_usd']
    )

    return agg_df, original_df


def prepare_model(agg_df: pd.DataFrame):
    model_df = agg_df.set_index('srch_id')
    model = NearestNeighbors(n_neighbors=6, algorithm='auto')  # default k+1
    model.fit(model_df)
    return model, model_df


def find_nearest_neighbors(model, model_df, target_id: int, k: int):
    if target_id not in model_df.index:
        raise ValueError(
            f"Target search ID {target_id} not found in aggregated data.")

    distances, indices = model.kneighbors(
        model_df.loc[[target_id]], n_neighbors=k + 1)
    neighbor_ids = model_df.iloc[indices[0]].index.tolist()

    if target_id in neighbor_ids:
        idx = neighbor_ids.index(target_id)
        neighbor_ids.pop(idx)
        distances = list(distances[0])
        distances.pop(idx)
    else:
        distances = list(distances[0])

    return neighbor_ids, distances


def recommend_hotels(original_df: pd.DataFrame, target_id: int, neighbor_ids: list):
    target_hotels = original_df[original_df['srch_id'] == target_id]
    target_prop_ids = set(target_hotels['prop_id'].values)

    neighbor_hotels = original_df[original_df['srch_id'].isin(neighbor_ids)]
    unseen_hotels = neighbor_hotels[~neighbor_hotels['prop_id'].isin(
        target_prop_ids)]
    unseen_hotels_unique = unseen_hotels.drop_duplicates(
        subset='prop_id').copy()

    # Ratio-based scoring with checks
    min_price = 10.0
    epsilon = 1e-6

    unseen_hotels_unique['prop_review_score'] = unseen_hotels_unique['prop_review_score'].clip(
        lower=0, upper=5)
    unseen_hotels_unique['adjusted_price'] = unseen_hotels_unique['price_usd'].clip(
        lower=min_price)
    unseen_hotels_unique['ratio_score'] = (
        unseen_hotels_unique['prop_review_score'] /
        (unseen_hotels_unique['adjusted_price'] + epsilon)
    )

    sorted_hotels = unseen_hotels_unique.sort_values(
        by='ratio_score', ascending=False)
    recommendations = sorted_hotels.head(len(target_hotels)).copy()
    recommendations['srch_id'] = target_id

    return recommendations[['srch_id', 'prop_id']]


def generate_recommendations_for_all(aggregated_path, original_path, k_neighbors=5):
    agg_df, original_df = load_data(aggregated_path, original_path)
    model, model_df = prepare_model(agg_df)
    unique_srch_ids = original_df['srch_id'].unique()

    all_recommendations = []

    for srch_id in tqdm(unique_srch_ids, desc="Recommending"):
        try:
            neighbor_ids, _ = find_nearest_neighbors(
                model, model_df, srch_id, k_neighbors)
            recommendations = recommend_hotels(
                original_df, srch_id, neighbor_ids)
            all_recommendations.append(recommendations)
        except Exception:
            continue  # Skip bad IDs silently for speed

    final_recommendations_df = pd.concat(
        all_recommendations, ignore_index=True)
    return final_recommendations_df[['srch_id', 'prop_id']]
