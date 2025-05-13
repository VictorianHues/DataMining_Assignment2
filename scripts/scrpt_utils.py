import os

from src.data_analysis import *
from src.plotting import *

def analyze_data(data_frame, name_extension, iqr_thresh=float('inf')):
    explore_dataset(data_frame)

    summary_path = os.path.join(os.path.dirname(__file__), "..", 'data', 'summary' + name_extension + '.csv')
    summary = summarize_dataframe(data_frame, iqr_thresh=iqr_thresh)
    summary.to_csv(summary_path, index=False)
    create_summary_table_visualization(summary, file_name='summary' + name_extension + '.png')


    plot_distributions(data_frame, file_name="distributions" + name_extension + ".png")

    for col in data_frame.columns:
        if pd.api.types.is_numeric_dtype(data_frame[col].dtype):
            plot_single_distribution(data_frame, col, xlabel=col, ylabel="Frequency",
                                    file_name=col + name_extension + ".png")
        else:
            print(f"Skipping non-numeric column: {col}")

