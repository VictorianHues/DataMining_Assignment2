import os

from src.csv_utils_basic import *
from src.data_cleaning import *
from src.plotting import *
from scrpt_utils import *

def main():
    read_file_path = os.path.join(os.path.dirname(__file__), "..", 'data', 'training_set_VU_DM.csv')
    write_file_path = os.path.join(os.path.dirname(__file__), "..", 'data', 'training_set_VU_DM_cleaned.csv')

    data_frame = read_csv(read_file_path)

    cleaned_data_frame = data_frame.copy()

    cleaned_data_frame = remove_duplicates(cleaned_data_frame)

    cleaned_data_frame = replace_iqr_outliers_with_na(cleaned_data_frame, 'prop_location_score1', iqr_thresh=1.5)
    cleaned_data_frame = replace_iqr_outliers_with_na(cleaned_data_frame, 'prop_location_score2', iqr_thresh=1.5)
    cleaned_data_frame = replace_iqr_outliers_with_na(cleaned_data_frame, 'prop_log_historical_price', iqr_thresh=1.5)
    cleaned_data_frame = replace_iqr_outliers_with_na(cleaned_data_frame, 'price_usd', iqr_thresh=1.5)
    cleaned_data_frame = replace_iqr_outliers_with_na(cleaned_data_frame, 'srch_length_of_stay', iqr_thresh=1.5)
    cleaned_data_frame = replace_iqr_outliers_with_na(cleaned_data_frame, 'srch_booking_window', iqr_thresh=1.5)
    #cleaned_data_frame = replace_iqr_outliers_with_na(cleaned_data_frame, 'srch_adults_count', iqr_thresh=1.5)
    #cleaned_data_frame = replace_iqr_outliers_with_na(cleaned_data_frame, 'srch_children_count', iqr_thresh=1.5)
    #cleaned_data_frame = replace_iqr_outliers_with_na(cleaned_data_frame, 'srch_room_count', iqr_thresh=1.5)
    
    analyze_data(cleaned_data_frame, name_extension='dataset_clean_')
    create_all_attribute_distributions(cleaned_data_frame, name_extension=os.path.join('clean', 'dataset_clean_'))

    dataframe_to_csv(cleaned_data_frame, write_file_path)




if __name__ == "__main__":
    main()