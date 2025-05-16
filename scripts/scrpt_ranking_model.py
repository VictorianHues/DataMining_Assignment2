import os
import lightgbm as lgb

from sklearn.model_selection import train_test_split

from src.csv_utils_basic import *
from src.plotting import *
from scrpt_utils import *

def GBDT_model():
    read_file_path = os.path.join(os.path.dirname(__file__), "..", 'data', 'training_set_VU_DM_engineered.csv')
    write_file_path = os.path.join(os.path.dirname(__file__), "..", 'data', 'training_set_VU_DM_GBDT.csv')
    write_submission_path = os.path.join(os.path.dirname(__file__), "..", 'submissions', 'submission_GBDT.csv')

    data_frame = read_csv(read_file_path)

    # Define features
    features = [
        'srch_id', 'prop_id',
        'prop_country_id', 'prop_starrating', 'prop_review_score',
        'prop_brand_bool', 'prop_location_score1', 'position',
        'price_usd', 'promotion_flag', 'srch_destination_id',
        'srch_length_of_stay', 'srch_booking_window',
        'srch_adults_count', 'srch_children_count', 'srch_room_count',
        'srch_saturday_night_bool',
        'sum_comp_rate', 'sum_comp_inv', 'log_price',
        'price_rank', 'review_rank', 'location_score_rank',
        'is_cheapest', 'is_best_rated',
        'hotel_booking_rate', 'avg_prop_review'
    ]

    X = data_frame[features]
    y = data_frame['relevance']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # For group information, you need to split srch_id as well
    srch_id_train = X_train['srch_id']
    srch_id_test = X_test['srch_id']
    group_train = srch_id_train.groupby(srch_id_train).transform('size').to_numpy()
    group_test = srch_id_test.groupby(srch_id_test).transform('size').to_numpy()

    # Create dataset
    lgb_train = lgb.Dataset(X_train, label=y_train, group=group_train)
    lgb_test = lgb.Dataset(X_test, label=y_test, group=group_test)

    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [5],
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbosity': -1,
    }

    model = lgb.train(params, lgb_train, num_boost_round=100, valid_sets=[lgb_test], valid_names=['test'])

    # Predict scores on test set
    data_frame['predicted_score'] = model.predict(data_frame[features])

    # Rank properties within each search
    data_frame['rank_within_search'] = data_frame.groupby('srch_id')['predicted_score'].rank(ascending=False)

    analyze_data(data_frame, name_extension='dataset_GBDT_')
    #create_all_attribute_distributions(data_frame, name_extension=os.path.join('GBDT', 'dataset_GBDT_'))

    dataframe_to_csv(data_frame, write_file_path)

    submission_df = data_frame[['srch_id', 'prop_id', 'rank_within_search']]
    submission_df = submission_df.sort_values(by=['srch_id', 'rank_within_search'])
    submission_df.drop(columns=['rank_within_search'], inplace=True)
    dataframe_to_csv(submission_df, write_submission_path)



if __name__ == "__main__":
    GBDT_model()