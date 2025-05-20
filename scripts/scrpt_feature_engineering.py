import os

from src.csv_utils_basic import *
from src.feature_engineering import *
from src.plotting import *
from scrpt_utils import *

def main():
    read_file_path = os.path.join(os.path.dirname(__file__), "..", 'data', 'training_set_VU_DM_imputed.csv')
    read_test_file_path = os.path.join(os.path.dirname(__file__), "..", 'data', 'test_set_VU_DM_imputed.csv')
    write_file_path = os.path.join(os.path.dirname(__file__), "..", 'data', 'training_set_VU_DM_engineered.csv')
    write_test_file_path = os.path.join(os.path.dirname(__file__), "..", 'data', 'test_set_VU_DM_engineered.csv')

    data_frame = read_csv(read_file_path)
    test_data_frame = read_csv(read_test_file_path)

    engineered_data_frame = data_frame.copy()
    test_engineered_data_frame = test_data_frame.copy()

    engineered_data_frame = sum_competitor_columns(engineered_data_frame)
    test_engineered_data_frame = sum_competitor_columns(test_engineered_data_frame)

    # Add global features
    engineered_data_frame, test_engineered_data_frame = add_global_features(engineered_data_frame, test_engineered_data_frame)

    # Convert to log price
    engineered_data_frame['log_price'] = np.log1p(engineered_data_frame['price_usd'])
    test_engineered_data_frame['log_price'] = np.log1p(test_engineered_data_frame['price_usd'])

    # Relative rank within search
    engineered_data_frame['price_rank'] = engineered_data_frame.groupby('srch_id')['price_usd'].rank()
    engineered_data_frame['review_rank'] = engineered_data_frame.groupby('srch_id')['prop_review_score'].rank(ascending=False)
    engineered_data_frame['location_score_rank'] = engineered_data_frame.groupby('srch_id')['prop_location_score1'].rank(ascending=False)

    test_engineered_data_frame['price_rank'] = test_engineered_data_frame.groupby('srch_id')['price_usd'].rank()
    test_engineered_data_frame['review_rank'] = test_engineered_data_frame.groupby('srch_id')['prop_review_score'].rank(ascending=False)
    test_engineered_data_frame['location_score_rank'] = test_engineered_data_frame.groupby('srch_id')['prop_location_score1'].rank(ascending=False)

    # Binary standout features
    engineered_data_frame['is_cheapest'] = engineered_data_frame.groupby('srch_id')['price_usd'].transform('min') == engineered_data_frame['price_usd']
    engineered_data_frame['is_best_rated'] = engineered_data_frame.groupby('srch_id')['prop_review_score'].transform('max') == engineered_data_frame['prop_review_score']

    test_engineered_data_frame['is_cheapest'] = test_engineered_data_frame.groupby('srch_id')['price_usd'].transform('min') == test_engineered_data_frame['price_usd']
    test_engineered_data_frame['is_best_rated'] = test_engineered_data_frame.groupby('srch_id')['prop_review_score'].transform('max') == test_engineered_data_frame['prop_review_score']
    
    # Encodes past hotel performance
    hotel_book_rate = engineered_data_frame.groupby('prop_id')['booking_bool'].mean()
    engineered_data_frame['hotel_booking_rate'] = engineered_data_frame['prop_id'].map(hotel_book_rate)

    prop_mean_score = engineered_data_frame.groupby('prop_id')['prop_review_score'].mean()
    engineered_data_frame['avg_prop_review'] = engineered_data_frame['prop_id'].map(prop_mean_score)

    #engineered_data_frame['relevance'] = engineered_data_frame['booking_bool'] * 5 + engineered_data_frame['click_bool']

    engineered_data_frame['price_z'] = engineered_data_frame.groupby('srch_id')['price_usd'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-5))
    test_engineered_data_frame['price_z'] = test_engineered_data_frame.groupby('srch_id')['price_usd'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-5))

    engineered_data_frame['price_percentile'] = engineered_data_frame.groupby('srch_id')['price_usd'].rank(pct=True)
    test_engineered_data_frame['price_percentile'] = test_engineered_data_frame.groupby('srch_id')['price_usd'].rank(pct=True)

    engineered_data_frame['is_top3_cheapest'] = engineered_data_frame['price_rank'] <= 3
    test_engineered_data_frame['is_top3_cheapest'] = test_engineered_data_frame['price_rank'] <= 3

    drop_cols = [
        'date_time', 'site_id', 
        'visitor_location_country_id', 'visitor_hist_starrating', 'visitor_hist_adr_usd',
        #'srch_id', 'prop_id',  # Keep these separately for submission
        #'click_bool', 'booking_bool',  # Keep only if training
        'random_bool', 'gross_bookings_usd',
        'orig_destination_distance', 'srch_query_affinity_score', 
        'prop_location_score2', 'prop_log_historical_price'
    ] + [col for col in engineered_data_frame.columns if col.startswith('comp')]

    drop_cols_test = [
        'date_time', 'site_id', 
        'visitor_location_country_id', 'visitor_hist_starrating', 'visitor_hist_adr_usd',
        #'srch_id', 'prop_id',  # Keep these separately for submission
        #'click_bool', 'booking_bool',  # Keep only if training
        'random_bool',
        'orig_destination_distance', 'srch_query_affinity_score', 
        'prop_location_score2', 'prop_log_historical_price'
    ] + [col for col in engineered_data_frame.columns if col.startswith('comp')]

    # Keep srch_id and prop_id in a separate index column for merging later
    #index_cols = data_frame[['srch_id', 'prop_id']]
    engineered_data_frame = engineered_data_frame.drop(columns=drop_cols)
    test_engineered_data_frame = test_engineered_data_frame.drop(columns=drop_cols_test)

    analyze_data(engineered_data_frame, name_extension='dataset_engineered_')
    create_all_attribute_distributions(engineered_data_frame, name_extension=os.path.join('engineered', 'dataset_engineered_'))

    analyze_data(test_engineered_data_frame, name_extension='dataset_test_engineered_')
    create_all_attribute_distributions(test_engineered_data_frame, name_extension=os.path.join('test_engineered', 'dataset_test_engineered_'))

    dataframe_to_csv(engineered_data_frame, write_file_path)
    dataframe_to_csv(test_engineered_data_frame, write_test_file_path)


if __name__ == "__main__":
    main()