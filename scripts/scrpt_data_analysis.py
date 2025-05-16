import os

from src.csv_utils_basic import *
from src.data_analysis import *
from src.plotting import *
from scrpt_utils import *

def main():
    read_file_path = os.path.join(os.path.dirname(__file__), "..", 'data', 'training_set_VU_DM.csv')

    data_frame = read_csv(read_file_path)
    analyze_data(data_frame, name_extension='dataset_', iqr_thresh=1.5)
    create_all_attribute_distributions(data_frame, name_extension=os.path.join('base', 'dataset_'))



if __name__ == "__main__":
    main()