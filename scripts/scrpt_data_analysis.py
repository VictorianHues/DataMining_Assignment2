import os

from src.csv_utils_basic import *
from src.data_analysis import *
from src.plotting import *
from scrpt_utils import *

def main():
    read_file_path = os.path.join(os.path.dirname(__file__), "..", 'data', 'training_set_VU_DM.csv')
    read_test_file_path = os.path.join(os.path.dirname(__file__), "..", 'data', 'test_set_VU_DM.csv')

    data_frame = read_csv(read_file_path)
    test_data_frame = read_csv(read_test_file_path)

    data_frame['log_price'] = np.log1p(data_frame['price_usd'])

    # booking_bool frequency
    booking_bool_freq = data_frame['booking_bool'].value_counts(normalize=True)
    print("Booking Bool Frequency:")
    print(booking_bool_freq)

    # click_bool frequency
    click_bool_freq = data_frame['click_bool'].value_counts(normalize=True)
    print("Click Bool Frequency:")
    print(click_bool_freq)


    create_correlation_heatmap(data_frame, file_name='correlation_heatmap.png')

    analyze_data(data_frame, name_extension='dataset_', iqr_thresh=1.5)
    create_all_attribute_distributions(data_frame, name_extension=os.path.join('base', 'dataset_'))

    analyze_data(test_data_frame, name_extension='dataset_test_', iqr_thresh=1.5)
    create_all_attribute_distributions(test_data_frame, name_extension=os.path.join('test_base', 'dataset_test_'))



if __name__ == "__main__":
    main()