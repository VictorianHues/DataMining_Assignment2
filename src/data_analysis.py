import os
import numpy as np
import numpy as np
import pandas as pd


def explore_dataset(dataframe):
    """
    Prints basic information about the DataFrame, including data types, summary statistics,
    and missing values.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame to explore.
    """
    print("~~~~~~~~~~~~~~~ Dataset Info ~~~~~~~~~~~~~~~")
    print(dataframe.info())
    print("\n~~~~~~~~~~~~~~~ Summary Statistics ~~~~~~~~~~~~~~~")
    print(dataframe.describe(include='all'))
    print("\n~~~~~~~~~~~~~~~ Missing Values ~~~~~~~~~~~~~~~")
    print(dataframe.isnull().sum())


def summarize_dataframe(df, iqr_thresh=float('inf')):
    summary = []

    for col in df.columns:
        col_data = df[col]
        col_dtype = col_data.dtype
        non_null_count = col_data.notnull().sum()
        missing_count = col_data.isnull().sum()
        unique_count = col_data.nunique(dropna=True)

        example_values = col_data.dropna().unique()[:2]
        example_values = [str(val) for val in example_values]

        min_val, max_val, mean_val, outlier_count = None, None, None, None

        if pd.api.types.is_numeric_dtype(col_dtype):
            if not set(col_data.dropna().unique()).issubset({0, 1}):  # skip binary fields
                min_val = np.round(col_data.min(), 5)
                max_val = np.round(col_data.max(), 5)
                mean_val = np.round(col_data.mean(), 5)

                if not np.isinf(iqr_thresh):
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - iqr_thresh * IQR
                    upper_bound = Q3 + iqr_thresh * IQR
                    outlier_count = ((col_data < lower_bound) | (col_data > upper_bound)).sum()

        summary.append({
            'Column': col,
            'Data Type': col_dtype,
            'Non-Null Count': non_null_count,
            'Missing Count': missing_count,
            'Unique Values': unique_count,
            'Min': min_val,
            'Max': max_val,
            'Mean': mean_val,
            'Outlier Count (IQR)': outlier_count,
            'Example Values': example_values
        })

    return pd.DataFrame(summary)