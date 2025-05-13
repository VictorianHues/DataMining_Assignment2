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

    # search_group = data_frame.groupby('srch_id')
    # mean_prices = search_group['price_usd'].mean()
    # search_counts = search_group.size()
    # search_summary = search_group.agg({
    #     'price_usd': ['mean', 'std', 'min', 'max'],
    #     'prop_review_score': ['mean', 'max'],
    #     'promotion_flag': 'sum',
    #     'click_bool': 'max',    # Whether any hotel was clicked
    #     'booking_bool': 'max'   # Whether any hotel was booked
    # })

    # # Flatten the multi-level columns
    # search_summary.columns = ['_'.join(col).strip() for col in search_summary.columns.values]
    # search_summary.reset_index(inplace=True)

    # plt.figure(figsize=(8, 4))
    # sns.histplot(search_summary['price_usd_mean'], bins=50, kde=True)
    # plt.title('Average Hotel Price per Search')
    # plt.xlabel('Average Price (USD)')
    # plt.ylabel('Number of Searches')
    # plt.show()

    # search_size = data_frame.groupby('srch_id').size()

    # plt.figure(figsize=(6, 4))
    # sns.histplot(search_size, bins=30)
    # plt.title("Number of Hotels per Search")
    # plt.xlabel("Hotel Count")
    # plt.ylabel("Search Count")
    # plt.show()

    # sns.boxplot(x='booking_bool_max', y='promotion_flag_sum', data=search_summary)
    # plt.title("Promotions vs. Booking")
    # plt.xlabel("Booking Occurred")
    # plt.ylabel("Number of Promoted Hotels in Search")
    # plt.show()



if __name__ == "__main__":
    main()