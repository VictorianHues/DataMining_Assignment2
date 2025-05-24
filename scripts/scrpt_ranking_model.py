import os
import pandas as pd

from src.csv_utils_basic import *
from src.plotting import *
from scrpt_utils import *
from src.gbdt_model import *

def get_features():
    return [
        'prop_country_id', 'prop_starrating', 'prop_review_score',
        'prop_brand_bool', 'prop_location_score1',
        'price_usd', 'promotion_flag', 'srch_destination_id',
        'srch_length_of_stay', 'srch_adults_count', 'srch_children_count', 'srch_room_count',
        'sum_comp_rate', 'sum_comp_inv', 'log_price',
        'price_rank', 'prop_review_score_rank', 'location_score_rank', 'value_score',
        'is_cheapest', 'is_best_rated', 'is_top3_cheapest', 'price_percentile', 'price_z',
        'is_top3_prop_review', 'relative_to_dest_price', 'prop_avg_price', 'avg_prop_review',
        'dest_avg_price', 'country_avg_star', 'avg_prop_star', 'price_per_adult',
        'visitor_location_country_id', 'visitor_hist_starrating', 'visitor_hist_adr_usd',
        'random_bool', 'orig_destination_distance', 'srch_query_affinity_score',
        'prop_location_score2', 'prop_log_historical_price', 'has_hist_rating',
        'is_weekend_search', 'is_weekday_search', 
        'price_vs_country_avg', 'price_country_zscore',
        'price_vs_location_avg', 'price_location_zscore', 
        'review_vs_location_avg', 'review_location_zscore'#,
        #'star_vs_location_avg', 'star_location_zscore'
    ]

def generate_submission(test_df, features, best_model, submission_file):
    test_df['predicted_score'] = best_model.predict(test_df[features])
    test_df['rank_within_search'] = test_df.groupby('srch_id')['predicted_score'].rank(method='first', ascending=False)
    submission_df = test_df.sort_values(['srch_id', 'rank_within_search'])[['srch_id', 'prop_id']]
    submission_df.to_csv(submission_file, index=False)
    print(f"Submission saved to: {submission_file}")

def print_feature_contributions(best_model, X_val, y_val, group_val, features):
    contrib = feature_contribution_ndcg(best_model, X_val, y_val, group_val, features)
    for feat, drop in sorted(contrib.items(), key=lambda x: -x[1]):
        print(f"{feat}: NDCG drop = {drop:.4f}")

def gbdt_model_param_grid():
    base_dir = os.path.dirname(__file__)
    #train_file = os.path.join(base_dir, "..", 'data', 'training_set_VU_DM_engineered.csv')
    #test_file = os.path.join(base_dir, "..", 'data', 'test_set_VU_DM_engineered.csv')
    #submission_file = os.path.join(base_dir, "..", 'submissions', 'submission_GBDT_5.csv')
    train_file = os.path.join(base_dir, "..", 'data', 'training_set_VU_DM_engineered_noimpute.csv')
    test_file = os.path.join(base_dir, "..", 'data', 'test_set_VU_DM_engineered_noimpute.csv')
    submission_file = os.path.join(base_dir, "..", 'submissions', 'submission_GBDT_noimpute.csv')

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    features = get_features()
    train_df = create_labels(train_df)
    X_train, y_train, group_train, X_val, y_val, group_val, val_idx = split_data(train_df, features)

    # param_grid = {
    #     'learning_rate': [0.01, 0.05, 0.1],
    #     'num_leaves': [32, 64, 128],
    #     'min_data_in_leaf': [25, 50, 100],
    #     'drop_rate': [0.01, 0.05, 0.1],
    # }

    param_grid = {
        'learning_rate': [0.05],
        'num_leaves': [32],
        'min_data_in_leaf': [25],
        'drop_rate': [0.1],
    }
    best_model = train_gbdt_iter(
        X_train, y_train, group_train, X_val, y_val, group_val,
        param_grid, results_csv="gbdt_grid_results_3.csv"
    )
    #generate_submission(test_df, features, best_model, submission_file)
    #print_feature_contributions(best_model, X_val, y_val, group_val, features)
    #save_feature_importance_plots(best_model)
    #plot_predicted_score_vs_relevance(best_model, X_val, y_val)
    #plot_feature_distribution_by_label(train_df, features, label_col='relevance')

    val_df = pd.DataFrame({
        'srch_id': train_df['srch_id'].values[val_idx],
        'prop_country_id': train_df['prop_country_id'].values[val_idx],
        'relevance': y_val,
        'predicted_score': best_model.predict(X_val)
    })
    
    bias_df = compute_ndcg_by_group(val_df, group_col='prop_country_id', k=5)
    print(bias_df.head())
    print(bias_df.tail())

    plot_ndcg_by_country(bias_df, k=5)

if __name__ == "__main__":
    gbdt_model_param_grid()