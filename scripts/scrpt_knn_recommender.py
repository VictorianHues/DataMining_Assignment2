from src.knn_recommender import *

aggregated_file = 'data/aggregated_data.csv'
original_file = 'data/test_set_VU_DM.csv'
k = 5

recommendations_df = generate_recommendations_for_all(
    aggregated_path=aggregated_file,
    original_path=original_file,
    k_neighbors=k
)

recommendations_df.to_csv("final_recommendations.csv", index=False)
print("Saved final_recommendations.csv")