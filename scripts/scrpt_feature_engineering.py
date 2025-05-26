import os

from src.csv_utils_basic import *
from src.feature_engineering import *
from src.plotting import *
from scrpt_utils import *

def engineer_features(df):
    df = sum_competitor_columns(df)
    
    df['log_price'] = np.log1p(df['price_usd'])

    df = add_ranking_features(df)
    df = add_binary_features(df)
    df = add_hotel_performance_features(df)
    df = add_price_features(df)
    df = add_time_aggregated_features(df)
    df = add_country_normalized_price(df)
    df = add_hotel_location_normalized_price(df)
    
    drop_cols = [ 'date_time' ] + [col for col in df.columns if col.startswith('comp')]
    df.drop(columns=[col for col in drop_cols if col in df.columns])

    return df

def process_and_save(read_path, write_path):
    df = read_csv(read_path)
    df = engineer_features(df)
    # analyze_data(df, name_extension=('dataset_test_engineered_' if is_test else 'dataset_engineered_'))
    # create_all_attribute_distributions(
    #     df, 
    #     name_extension=os.path.join(('test_engineered' if is_test else 'engineered'), 
    #                                ('dataset_test_engineered_' if is_test else 'dataset_engineered_'))
    # )
    dataframe_to_csv(df, write_path)

def main():
    base_dir = os.path.dirname(__file__)
    #read_file_path = os.path.join(base_dir, "..", 'data', 'training_set_VU_DM_imputed.csv')
    #read_test_file_path = os.path.join(base_dir, "..", 'data', 'test_set_VU_DM_imputed.csv')
    #write_file_path = os.path.join(base_dir, "..", 'data', 'training_set_VU_DM_engineered.csv')
    #write_test_file_path = os.path.join(base_dir, "..", 'data', 'test_set_VU_DM_engineered.csv')

    read_file_path = os.path.join(base_dir, "..", 'data', 'training_set_VU_DM_cleaned.csv')
    read_test_file_path = os.path.join(base_dir, "..", 'data', 'test_set_VU_DM_cleaned.csv')
    write_file_path = os.path.join(base_dir, "..", 'data', 'training_set_VU_DM_engineered_noimpute.csv')
    write_test_file_path = os.path.join(base_dir, "..", 'data', 'test_set_VU_DM_engineered_noimpute.csv')

    process_and_save(read_file_path, write_file_path)
    process_and_save(read_test_file_path, write_test_file_path)

if __name__ == "__main__":
    main()