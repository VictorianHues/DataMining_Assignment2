from src.knn_recommender import *

# === Configuration ===
AGGREGATED_DATA_PATH = 'data/aggregated_data.csv'
FULL_DATA_PATH = 'data/training_set_VU_DM.csv'
OUTPUT_PATH = 'data/recommendations.csv'

K_NEIGHBORS = 5         # total number of neighbors (including self)
TOP_N_RECOMMENDATIONS = 10  # top N unseen properties to recommend

CLICK_WEIGHT = 0.3
BOOK_WEIGHT = 0.7

# === Run ===
if __name__ == "__main__":
    recommend_for_all_users(
        agg_path=AGGREGATED_DATA_PATH,
        full_data_path=FULL_DATA_PATH,
        output_path=OUTPUT_PATH,
        k_neighbors=K_NEIGHBORS,
        top_n=TOP_N_RECOMMENDATIONS,
        click_weight=CLICK_WEIGHT,
        book_weight=BOOK_WEIGHT
    )
