import os

from src.csv_utils_basic import *
from src.feature_engineering import *
from src.plotting import *
from scrpt_utils import *

def main():
    read_file_path = os.path.join(os.path.dirname(__file__), "..", 'data', 'training_set_VU_DM_imputed.csv')
    write_file_path = os.path.join(os.path.dirname(__file__), "..", 'data', 'training_set_VU_DM_engineered.csv')

    data_frame = read_csv(read_file_path)

    engineered_data_frame = data_frame.copy()

    engineered_data_frame = sum_competitor_columns(engineered_data_frame)

    
    # Convert to log price
    engineered_data_frame['log_price'] = np.log1p(engineered_data_frame['price_usd'])

    # Relative rank within search
    engineered_data_frame['price_rank'] = engineered_data_frame.groupby('srch_id')['price_usd'].rank()
    engineered_data_frame['review_rank'] = engineered_data_frame.groupby('srch_id')['prop_review_score'].rank(ascending=False)
    engineered_data_frame['location_score_rank'] = engineered_data_frame.groupby('srch_id')['prop_location_score1'].rank(ascending=False)

    # Binary standout features
    engineered_data_frame['is_cheapest'] = engineered_data_frame.groupby('srch_id')['price_usd'].transform('min') == engineered_data_frame['price_usd']
    engineered_data_frame['is_best_rated'] = engineered_data_frame.groupby('srch_id')['prop_review_score'].transform('max') == engineered_data_frame['prop_review_score']
    
    # Encodes past hotel performance
    hotel_book_rate = engineered_data_frame.groupby('prop_id')['booking_bool'].mean()
    engineered_data_frame['hotel_booking_rate'] = engineered_data_frame['prop_id'].map(hotel_book_rate)

    prop_mean_score = engineered_data_frame.groupby('prop_id')['prop_review_score'].mean()
    engineered_data_frame['avg_prop_review'] = engineered_data_frame['prop_id'].map(prop_mean_score)

    engineered_data_frame['relevance'] = engineered_data_frame['booking_bool'] * 5 + engineered_data_frame['click_bool']


    drop_cols = [
        'date_time', 'site_id', 
        'visitor_location_country_id', 'visitor_hist_starrating', 'visitor_hist_adr_usd',
        #'srch_id', 'prop_id',  # Keep these separately for submission
        #'click_bool', 'booking_bool',  # Keep only if training
        'random_bool', 'gross_bookings_usd',
        'orig_destination_distance', 'srch_query_affinity_score', 
        'prop_location_score2', 'prop_log_historical_price'
    ] + [col for col in engineered_data_frame.columns if col.startswith('comp')]

    # Keep srch_id and prop_id in a separate index column for merging later
    #index_cols = data_frame[['srch_id', 'prop_id']]
    engineered_data_frame = engineered_data_frame.drop(columns=drop_cols)

    analyze_data(engineered_data_frame, name_extension='dataset_engineered_')
    create_all_attribute_distributions(engineered_data_frame, name_extension=os.path.join('engineered', 'dataset_engineered_'))

    dataframe_to_csv(engineered_data_frame, write_file_path)


if __name__ == "__main__":
    main()