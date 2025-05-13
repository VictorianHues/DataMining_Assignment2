import os
import matplotlib.pyplot as plt
import seaborn as sns

from src.csv_utils_basic import *
from src.data_analysis import *
from src.data_cleaning import *
from src.plotting import *
from scrpt_utils import *

def main():
    read_file_path = os.path.join(os.path.dirname(__file__), "..", 'data', 'training_set_VU_DM.csv')
    write_file_path = os.path.join(os.path.dirname(__file__), "..", 'data', 'training_set_VU_DM_cleaned.csv')

    data_frame = read_csv(read_file_path)

    cleaned_data_frame = data_frame.copy()
    

    drop_cols = [
        'date_time', 'date_time', 'site_id', 
        'visitor_location_country_id', 'visitor_hist_starrating', 'visitor_hist_adr_usd',
        'srch_id', 'prop_id',  # Keep these separately for submission
        #'click_bool', 'booking_bool',  # Keep only if training
        'random_bool', 'gross_bookings_usd'
    ] + [col for col in cleaned_data_frame.columns if col.startswith('comp')]

    # Keep srch_id and prop_id in a separate index column for merging later
    index_cols = data_frame[['srch_id', 'prop_id']]

    cleaned_data_frame = cleaned_data_frame.drop(columns=drop_cols)

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

    analyze_data(cleaned_data_frame, name_extension='training_set_VU_DM_CLEAN')




if __name__ == "__main__":
    main()