import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder


def impute_categoricals_mode(data_frame, cat_cols):
    """
    Imputes missing values in categorical columns using the mode (most frequent value).
    This method replaces NaN values with the most common value in each column.

    Parameters:
        data_frame (pd.DataFrame): The DataFrame containing categorical columns.
        cat_cols (list): List of categorical column names.
    
    Returns:
        pd.DataFrame: The DataFrame with missing values in categorical columns imputed.
    """
    print(f"Imputing {len(cat_cols)} categorical columns using mode")

    df_cat = data_frame[cat_cols].copy()
    for col in df_cat:
        df_cat[col] = df_cat[col].fillna(df_cat[col].mode().iloc[0])
    data_frame[cat_cols] = df_cat
    return data_frame

def impute_categoricals_knn(data_frame, cat_cols, k=5, sample_frac=0.25, random_state=None):
    """
    Impute missing values in categorical columns using KNN imputation based on a sample of the dataset.

    Parameters:
        data_frame (pd.DataFrame): The DataFrame containing categorical columns.
        cat_cols (list): List of categorical column names.
        k (int): Number of neighbors to use for imputation.
        sample_frac (float): Fraction of the data to use for fitting the imputer.
        random_state (int, optional): Random state for reproducibility.

    Returns:
        pd.DataFrame: The DataFrame with missing values in categorical columns imputed.
    """
    print(f"Imputing {len(cat_cols)} categorical columns using KNN with k={k} and sample fraction={sample_frac}")

    df_cat = data_frame[cat_cols].copy()
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    encoded = encoder.fit_transform(df_cat)

    # Sample the data for fitting the imputer
    sample_idx = df_cat.sample(frac=sample_frac, random_state=random_state).index
    sample_encoded = encoded[sample_idx, :]

    imputer = KNNImputer(n_neighbors=k)
    imputer.fit(sample_encoded)
    imputed = imputer.transform(encoded)

    df_cat[:] = encoder.inverse_transform(imputed)
    for col in df_cat.columns:
        df_cat[col] = df_cat[col].astype('category')
    data_frame[cat_cols] = df_cat
    return data_frame

def impute_numericals_median(data_frame, num_cols):
    """
    Imputes missing values in numeric columns using the median.
    This method replaces NaN values with the median value of each column.

    Parameters:
        data_frame (pd.DataFrame): The DataFrame containing numeric columns.
        num_cols (list): List of numeric column names.
    
    Returns:
        pd.DataFrame: The DataFrame with missing values in numeric columns imputed.
    """
    print(f"Imputing {len(num_cols)} numeric columns using median")
    
    for col in num_cols:
        data_frame[col] = data_frame[col].fillna(data_frame[col].median())
    return data_frame

def impute_numericals_knn(data_frame, num_cols, k=5, sample_frac=0.25, random_state=None):
    """
    Imputes missing values in numeric columns using KNN imputation based on a sample of the dataset.

    Parameters:
        data_frame (pd.DataFrame): The DataFrame containing numeric columns.
        num_cols (list): List of numeric column names.
        k (int): Number of neighbors to use for imputation.
        sample_frac (float): Fraction of the data to use for fitting the imputer.
        random_state (int, optional): Random state for reproducibility.

    Returns:
        pd.DataFrame: The DataFrame with missing values in numeric columns imputed.
    """
    print(f"Imputing {len(num_cols)} numeric columns using KNN with k={k} and sample fraction={sample_frac}")

    df_num = data_frame[num_cols]
    # Sample the data for fitting the imputer
    sample_idx = df_num.sample(frac=sample_frac, random_state=random_state).index
    sample_data = df_num.loc[sample_idx]

    imputer = KNNImputer(n_neighbors=k)
    imputer.fit(sample_data)
    data_frame[num_cols] = imputer.transform(df_num)
    return data_frame
