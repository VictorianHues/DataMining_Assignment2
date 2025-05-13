import os
import matplotlib.pyplot as plt
import seaborn as sns

from src.csv_utils_basic import *
from src.data_analysis import *
from src.plotting import *
from scrpt_utils import *

def main():
    read_file_path = os.path.join(os.path.dirname(__file__), "..", 'data', 'training_set_VU_DM.csv')
    write_file_path = os.path.join(os.path.dirname(__file__), "..", 'data', 'training_set_VU_DM_cleaned.csv')

    data_frame = read_csv(read_file_path)
    analyze_data(data_frame, name_extension='training_set_VU_DM', iqr_thresh=1.5)

    



if __name__ == "__main__":
    main()