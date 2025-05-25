from src.knn_recommender import generate_recommendations_for_all
import pandas as pd

# File paths
AGGREGATED_DATA_PATH = 'data/aggregated_data.csv'
ORIGINAL_DATA_PATH = 'data/test_set_VU_DM_engineered.csv'
OUTPUT_PATH = 'data/knn_recommendations.csv'

# Parameters
K_NEIGHBORS = 5

if __name__ == "__main__":
    print("📥 Generating recommendations...")
    recommendations = generate_recommendations_for_all(
        aggregated_path=AGGREGATED_DATA_PATH,
        original_path=ORIGINAL_DATA_PATH,
        k_neighbors=K_NEIGHBORS
    )

    print(f"💾 Saving to {OUTPUT_PATH}")
    recommendations.to_csv(OUTPUT_PATH, index=False)
    print("✅ Done.")
