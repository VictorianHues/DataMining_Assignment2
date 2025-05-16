import os

from src.data_analysis import *
from src.plotting import *

def analyze_data(data_frame, name_extension, iqr_thresh=float('inf')):
    explore_dataset(data_frame)

    summary_path = os.path.join(os.path.dirname(__file__), "..", 'data', name_extension + 'summary.csv')
    summary = summarize_dataframe(data_frame, iqr_thresh=iqr_thresh)
    summary.to_csv(summary_path, index=False)
    create_summary_table_visualization(summary, file_name= name_extension + 'summary.png')

    plot_distributions(data_frame, file_name=name_extension + "distributions.png")


def create_all_attribute_distributions(data_frame, name_extension):
    for col in data_frame.columns:
        if pd.api.types.is_numeric_dtype(data_frame[col].dtype):
            plot_single_distribution(data_frame, col, xlabel=col, ylabel="Frequency",
                                    file_name=name_extension + col + ".png")
        else:
            print(f"Skipping non-numeric column: {col}")


def convert_to_categorical(data_frame, col_name):
    """
    Converts a column in the DataFrame to categorical type.

    Parameters:
        data_frame (pd.DataFrame): The DataFrame containing the column to convert.
        col_name (str): The name of the column to convert.

    Returns:
        pd.DataFrame: The DataFrame with the specified column converted to categorical type.
    """
    data_frame[col_name] = data_frame[col_name].astype('category')
    return data_frame

def convert_categorical_to_numeric(data_frame, col_name):
    """
    Converts a categorical column in the DataFrame to numeric type.

    Parameters:
        data_frame (pd.DataFrame): The DataFrame containing the column to convert.
        col_name (str): The name of the column to convert.

    Returns:
        pd.DataFrame: The DataFrame with the specified column converted to numeric type.
    """
    data_frame[col_name] = data_frame[col_name].cat.codes
    return data_frame