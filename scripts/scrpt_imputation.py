import os

from src.csv_utils_basic import *
from src.imputation import *
from src.plotting import *
from scrpt_utils import *

def main():
    read_file_path = os.path.join(os.path.dirname(__file__), "..", 'data', 'training_set_VU_DM_cleaned.csv')
    write_file_path = os.path.join(os.path.dirname(__file__), "..", 'data', 'training_set_VU_DM_imputed.csv')

    imputed_data_frame = read_csv(read_file_path)
    
    imputed_data_frame = impute_numericals_median(imputed_data_frame, ['prop_log_historical_price'])
    imputed_data_frame = impute_numericals_median(imputed_data_frame, ['price_usd'])
    imputed_data_frame = impute_numericals_median(imputed_data_frame, ['srch_booking_window'])
    #imputed_data_frame = impute_numericals_knn(imputed_data_frame, ['prop_log_historical_price'])
    #imputed_data_frame = impute_numericals_knn(imputed_data_frame, ['price_usd'])
    #imputed_data_frame = impute_numericals_knn(imputed_data_frame, ['srch_booking_window'])


    imputed_data_frame = convert_to_categorical(imputed_data_frame, 'prop_review_score')
    imputed_data_frame = convert_to_categorical(imputed_data_frame, 'srch_length_of_stay')

    #imputed_data_frame = impute_categoricals_mode(imputed_data_frame, ['prop_review_score'])
    #imputed_data_frame = impute_categoricals_mode(imputed_data_frame, ['srch_length_of_stay'])
    imputed_data_frame = impute_categoricals_knn(imputed_data_frame, ['prop_review_score'])
    imputed_data_frame = impute_categoricals_knn(imputed_data_frame, ['srch_length_of_stay'])

    imputed_data_frame = convert_categorical_to_numeric(imputed_data_frame, 'prop_review_score')
    imputed_data_frame = convert_categorical_to_numeric(imputed_data_frame, 'srch_length_of_stay')

    analyze_data(imputed_data_frame, name_extension='dataset_imputed_')
    create_all_attribute_distributions(imputed_data_frame, name_extension=os.path.join('imputed', 'dataset_imputed_'))

    dataframe_to_csv(imputed_data_frame, write_file_path)


if __name__ == "__main__":
    main()