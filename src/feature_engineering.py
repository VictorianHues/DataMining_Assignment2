import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder


def sum_competitor_columns(data_frame):
    data_frame['sum_comp_rate'] = data_frame[[f'comp{i}_rate' for i in range(1, 9)]].sum(axis=1, skipna=True)
    data_frame['sum_comp_inv'] = data_frame[[f'comp{i}_inv' for i in range(1, 9)]].sum(axis=1, skipna=True)


    data_frame['sum_comp_rate'] = data_frame['sum_comp_rate'].fillna(0.0)
    data_frame['sum_comp_inv'] = data_frame['sum_comp_inv'].fillna(0.0)

    return data_frame


def add_global_features(train_df, test_df):
    """
    Add global features to the training and test dataframes.
    Global features include booking rate, click rate, average price,
    and average review score for properties, as well as average price
    for destinations and average star rating for countries.

    Args:
        train_df (pd.DataFrame): The training dataframe.
        test_df (pd.DataFrame): The test dataframe.
    Returns:
        train_df (pd.DataFrame): The training dataframe with global features.
        test_df (pd.DataFrame): The test dataframe with global features.
    """

    # Calculate global features
    global_features = train_df.groupby('prop_id').agg(
        prop_booking_rate=('booking_bool', 'mean'),
        prop_click_rate=('click_bool', 'mean'),
        prop_avg_price=('price_usd', 'mean'),
        prop_avg_review_score=('prop_review_score', 'mean')
    ).reset_index()

    # Calculate destination and country features
    dest_agg = train_df.groupby('srch_destination_id').agg(
        dest_avg_price=('price_usd', 'mean')
    ).reset_index()

    # Calculate country features
    country_agg = train_df.groupby('prop_country_id').agg(
        country_avg_star=('prop_starrating', 'mean')
    ).reset_index()

    # Merge global features with train and test dataframes
    train_df = train_df.merge(global_features, on='prop_id', how='left')
    test_df = test_df.merge(global_features, on='prop_id', how='left')

    # Merge destination and country features
    train_df = train_df.merge(dest_agg, on='srch_destination_id', how='left')
    test_df = test_df.merge(dest_agg, on='srch_destination_id', how='left')

    # Merge country features
    train_df = train_df.merge(country_agg, on='prop_country_id', how='left')
    test_df = test_df.merge(country_agg, on='prop_country_id', how='left')

    return train_df, test_df