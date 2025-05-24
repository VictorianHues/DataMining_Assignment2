import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder


def sum_competitor_columns(data_frame):
    """
    Sums the competitor columns in the DataFrame.
    The columns to be summed are named 'comp1_rate', 'comp2_rate', ..., 'comp8_rate' and
    'comp1_inv', 'comp2_inv', ..., 'comp8_inv'.
    The resulting sums are stored in new columns 'sum_comp_rate' and 'sum_comp_inv'.
    The NaN values in the summed columns are filled with 0.0.

    Parameters:
        data_frame (pd.DataFrame): The DataFrame containing the competitor columns.

    Returns:
        pd.DataFrame: The DataFrame with the summed columns added.
    """
    data_frame['sum_comp_rate'] = data_frame[[f'comp{i}_rate' for i in range(1, 9)]].sum(axis=1, skipna=True)
    data_frame['sum_comp_inv'] = data_frame[[f'comp{i}_inv' for i in range(1, 9)]].sum(axis=1, skipna=True)


    data_frame['sum_comp_rate'] = data_frame['sum_comp_rate'].fillna(0.0)
    data_frame['sum_comp_inv'] = data_frame['sum_comp_inv'].fillna(0.0)

    return data_frame

def add_ranking_features(df):
    """
    Adds ranking features to the DataFrame.
    The features include:
    - price_rank: Rank of the price in ascending order within each search ID.
    - location_score_rank: Rank of the property location score in descending order within each search ID.
    - prop_star_rank: Rank of the property star rating in descending order within each search ID.
    - prop_review_score_rank: Rank of the property review score in descending order within each search ID.
    The ranks are calculated using the pandas rank method.
    The ranks are assigned within each search ID group.
    The ranks are assigned in ascending order for price and descending order for the other features.
    Parameters:
        df (pd.DataFrame): The DataFrame containing the features.
    Returns:
        pd.DataFrame: The DataFrame with the ranking features added.
    """
    df['price_rank'] = df.groupby('srch_id')['price_usd'].rank()
    df['location_score_rank'] = df.groupby('srch_id')['prop_location_score1'].rank(ascending=False)
    df['prop_star_rank'] = df.groupby('srch_id')['prop_starrating'].rank(ascending=False)
    df['prop_review_score_rank'] = df.groupby('srch_id')['prop_review_score'].rank(ascending=False)

    return df

def add_binary_features(df):
    """
    Adds binary features to the DataFrame.
    The features include:
    - is_cheapest: Indicates if the property is the cheapest within each search ID.
    - is_best_rated: Indicates if the property has the best review score within each search ID.
    """
    df['is_cheapest'] = df.groupby('srch_id')['price_usd'].transform('min') == df['price_usd']
    df['is_best_rated'] = df.groupby('srch_id')['prop_review_score'].transform('max') == df['prop_review_score']
    return df

def add_hotel_performance_features(df):
    """
    Adds hotel performance features to the DataFrame.
    The features include:
    - avg_prop_review: Average review score of the property across all searches.
    """
    prop_mean_score = df.groupby('prop_id')['prop_review_score'].mean()
    df['avg_prop_review'] = df['prop_id'].map(prop_mean_score)
    prop_mean_star = df.groupby('prop_id')['prop_starrating'].mean()
    df['avg_prop_star'] = df['prop_id'].map(prop_mean_star)
    dest_avg_price = df.groupby('srch_destination_id')['price_usd'].mean()
    df['dest_avg_price'] = df['srch_destination_id'].map(dest_avg_price)
    country_avg_star = df.groupby('visitor_location_country_id')['prop_starrating'].mean()
    df['country_avg_star'] = df['visitor_location_country_id'].map(country_avg_star)
    prop_avg_price = df.groupby('prop_id')['price_usd'].mean()
    df['prop_avg_price'] = df['prop_id'].map(prop_avg_price)

    df['value_score'] = df['prop_review_score'] / np.log1p(df['price_usd'])
    df['has_hist_rating'] = df['visitor_hist_starrating'].notnull().astype(int)
    return df

def add_price_features(df):
    """
    Adds price-related features to the DataFrame.
    The features include:
    - price_z: Z-score of the price within each search ID.
    - price_percentile: Percentile rank of the price within each search ID.
    - is_top3_cheapest: Indicates if the property is among the top 3 cheapest within each search ID.

    """
    df['price_z'] = df.groupby('srch_id')['price_usd'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-5))
    df['price_percentile'] = df.groupby('srch_id')['price_usd'].rank(pct=True)
    df['is_top3_cheapest'] = df['price_rank'] <= 3
    df['is_top3_prop_review'] = df['prop_review_score_rank'] <= 3
    df['relative_to_dest_price'] = df['price_usd'] / df['dest_avg_price']
    df['price_per_adult'] = df['price_usd'] / (df['srch_adults_count'] + 1e-5)
    return df

def add_time_aggregated_features(df):
    df['is_weekend_search'] = df['date_time'].dt.dayofweek >= 5
    df['is_weekday_search'] = df['date_time'].dt.dayofweek < 5
    df['search_hour'] = df['date_time'].dt.hour
    return df

def add_country_normalized_price(df):
    """
    Adds features that normalize hotel price by country.
    - price_vs_country_avg: Ratio of price to average price in the visitor's country.
    - price_country_zscore: Z-score of price within the visitor's country.
    """
    country_avg_price = df.groupby('visitor_location_country_id')['price_usd'].transform('mean')
    country_std_price = df.groupby('visitor_location_country_id')['price_usd'].transform('std').fillna(1)
    df['price_vs_country_avg'] = df['price_usd'] / (country_avg_price + 1e-5)
    df['price_country_zscore'] = (df['price_usd'] - country_avg_price) / (country_std_price + 1e-5)
    return df

def add_hotel_location_normalized_price(df):
    """
    Adds features that normalize hotel price by location.
    - price_vs_location_avg: Ratio of price to average price in the location.
    - price_location_zscore: Z-score of price within the location.
    """
    location_avg_price = df.groupby('prop_country_id')['price_usd'].transform('mean')
    location_std_price = df.groupby('prop_country_id')['price_usd'].transform('std').fillna(1)
    df['price_vs_location_avg'] = df['price_usd'] / (location_avg_price + 1e-5)
    df['price_location_zscore'] = (df['price_usd'] - location_avg_price) / (location_std_price + 1e-5)

    # Review score normalization
    location_avg_review = df.groupby('prop_country_id')['prop_review_score'].transform('mean')
    location_std_review = df.groupby('prop_country_id')['prop_review_score'].transform('std').fillna(1)
    df['review_vs_location_avg'] = df['prop_review_score'] / (location_avg_review + 1e-5)
    df['review_location_zscore'] = (df['prop_review_score'] - location_avg_review) / (location_std_review + 1e-5)

    # Star rating normalization
    location_avg_star = df.groupby('prop_country_id')['prop_starrating'].transform('mean')
    location_std_star = df.groupby('prop_country_id')['prop_starrating'].transform('std').fillna(1)
    df['star_vs_location_avg'] = df['prop_starrating'] / (location_avg_star + 1e-5)
    df['star_location_zscore'] = (df['prop_starrating'] - location_avg_star) / (location_std_star + 1e-5)


    return df