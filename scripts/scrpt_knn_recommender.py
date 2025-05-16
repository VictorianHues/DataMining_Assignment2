from src.knn_recommender import recommend_for_all_users

# File paths
aggregated_data_path = 'data/aggregated_data.csv'
full_training_data_path = 'data/training_set_VU_DM.csv'
output_file_path = 'output/recommendations.csv'

# Parameters
K_NEIGHBORS = 5
TOP_N = 10
CLICK_WEIGHT = 1.0
BOOK_WEIGHT = 4.0
BATCH_SIZE = 1000

if __name__ == "__main__":
    recommend_for_all_users(
        agg_path=aggregated_data_path,
        full_data_path=full_training_data_path,
        output_path=output_file_path,
        k_neighbors=K_NEIGHBORS,
        top_n=TOP_N,
        click_weight=CLICK_WEIGHT,
        book_weight=BOOK_WEIGHT,
        batch_size=BATCH_SIZE
    )
