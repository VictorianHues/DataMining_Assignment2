import numpy as np
import pandas as pd

def remove_duplicates(data_frame):
    """
    Removes duplicate rows from the DataFrame.

    Parameters:
        data_frame (pd.DataFrame): The DataFrame from which to remove duplicates.

    Returns:
        pd.DataFrame: The DataFrame with duplicates removed.
    """
    return data_frame.drop_duplicates()


def replace_iqr_outliers_with_na(data_frame, col_name, iqr_thresh=1.5):
    """
    Replaces outliers in the DataFrame with NA based on the IQR method.

    Parameters:
        data_frame (pd.DataFrame): The DataFrame in which to replace outliers.
        col_name (str): The name of the column to check for outliers.
        iqr_thresh (float): The IQR threshold for identifying outliers.

    Returns:
        pd.DataFrame: The DataFrame with outliers replaced by NA.
    """
    Q1 = data_frame[col_name].quantile(0.25)
    Q3 = data_frame[col_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_thresh * IQR
    upper_bound = Q3 + iqr_thresh * IQR

    data_frame[col_name] = data_frame[col_name].mask(
        (data_frame[col_name] < lower_bound) | (data_frame[col_name] > upper_bound), pd.NA
    )
    return data_frame
