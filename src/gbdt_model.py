import lightgbm as lgb
import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import GroupKFold
from sklearn.metrics import ndcg_score
from sklearn.utils import shuffle

def create_labels(train_df):
    train_df['relevance'] = train_df['booking_bool'] * 2 + train_df['click_bool']
    train_df['relevance'] = train_df['relevance'].astype(int)
    return train_df

def split_data(train_df, features):
    """
    Splits the data into training and validation sets using GroupKFold on 'srch_id'.
    """
    gkf = GroupKFold(n_splits=5)
    train_idx, val_idx = next(gkf.split(train_df[features], train_df['relevance'], groups=train_df['srch_id']))
    X_train = train_df.iloc[train_idx][features]
    y_train = train_df.iloc[train_idx]['relevance']
    srch_train = train_df.iloc[train_idx]['srch_id']
    X_val = train_df.iloc[val_idx][features]
    y_val = train_df.iloc[val_idx]['relevance']
    srch_val = train_df.iloc[val_idx]['srch_id']
    group_train = srch_train.groupby(srch_train).size().to_numpy()
    group_val = srch_val.groupby(srch_val).size().to_numpy()
    return X_train, y_train, group_train, X_val, y_val, group_val, val_idx

def train_gbdt(X_train, y_train, group_train, X_val, y_val, group_val, params):
    lgb_train = lgb.Dataset(X_train, label=y_train, group=group_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, group=group_val)
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=200,
        valid_sets=[lgb_val],
        valid_names=['val']
    )
    return model

def train_gbdt_iter(X_train, y_train, group_train, X_val, y_val, group_val, param_grid, val_df=None, train_df=None, results_csv="gbdt_grid_results.csv"):
    """
    Grid search for GBDT hyperparameters using NDCG@5.
    """
    best_ndcg = -1
    best_model = None
    best_params = None
    results = []

    for lr, nl, mdl, dr in itertools.product(
        param_grid['learning_rate'],
        param_grid['num_leaves'],
        param_grid['min_data_in_leaf'],
        param_grid['drop_rate']
    ):
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [5],
            'learning_rate': lr,
            'num_leaves': nl,
            'min_data_in_leaf': mdl,
            'boosting_type': 'dart',
            'drop_rate': dr,
            'verbosity': -1
        }
        model = train_gbdt(X_train, y_train, group_train, X_val, y_val, group_val, params)
        val_pred = model.predict(X_val)
        if val_df is None:
            val_df_eval = pd.DataFrame(X_val.copy())
            val_df_eval['relevance'] = y_val.values
            val_df_eval['srch_id'] = X_val['srch_id'].values if 'srch_id' in X_val.columns else group_val.repeat(group_val)
            val_df_eval['predicted_score'] = val_pred
        else:
            val_df_eval = val_df.copy()
            val_df_eval['predicted_score'] = val_pred
        ndcg_val = val_df_eval.groupby('srch_id').apply(ndcg_per_group).mean()

        train_pred = model.predict(X_train)
        if train_df is None:
            train_df_eval = pd.DataFrame(X_train.copy())
            train_df_eval['relevance'] = y_train.values
            train_df_eval['srch_id'] = X_train['srch_id'].values if 'srch_id' in X_train.columns else group_train.repeat(group_train)
            train_df_eval['predicted_score'] = train_pred
        else:
            train_df_eval = train_df.copy()
            train_df_eval['predicted_score'] = train_pred
        ndcg_train = train_df_eval.groupby('srch_id').apply(ndcg_per_group).mean()

        print(f"Params: {params}")
        print(f"Train NDCG@5: {ndcg_train:.5f} | Validation NDCG@5: {ndcg_val:.5f}")

        results.append({
            'learning_rate': lr,
            'num_leaves': nl,
            'min_data_in_leaf': mdl,
            'drop_rate': dr,
            'ndcg_train': ndcg_train,
            'ndcg_val': ndcg_val
        })

        if ndcg_val > best_ndcg:
            best_ndcg = ndcg_val
            best_model = model
            best_params = params

    pd.DataFrame(results).to_csv(results_csv, index=False)
    print(f"\nAll results saved to {results_csv}")
    print(f"\nBest params: {best_params}")
    print(f"Best Validation NDCG@5: {best_ndcg:.5f}")
    return best_model

def ndcg_per_group(group):
    true = group['relevance'].values.reshape(1, -1)
    pred = group['predicted_score'].values.reshape(1, -1)
    return ndcg_score(true, pred, k=5)

def feature_contribution_ndcg(model, X_val, y_val, group_val, features, k=5):
    """
    Estimate feature importance by NDCG drop after permutation.
    """
    df_val = X_val.copy()
    df_val['relevance'] = y_val.values
    if 'srch_id' not in df_val.columns:
        srch_ids = np.concatenate([[i]*g for i, g in enumerate(group_val)])
        df_val['srch_id'] = srch_ids
    df_val['predicted_score'] = model.predict(X_val)
    baseline_ndcg = df_val.groupby('srch_id').apply(ndcg_per_group).mean()
    contributions = {}
    for feat in features:
        X_val_shuffled = X_val.copy()
        X_val_shuffled[feat] = shuffle(X_val_shuffled[feat].values, random_state=42)
        df_val_shuffled = X_val_shuffled.copy()
        df_val_shuffled['relevance'] = y_val.values
        df_val_shuffled['srch_id'] = df_val['srch_id'].values
        df_val_shuffled['predicted_score'] = model.predict(X_val_shuffled)
        ndcg = df_val_shuffled.groupby('srch_id').apply(ndcg_per_group).mean()
        contributions[feat] = baseline_ndcg - ndcg
    return contributions

def compute_ndcg_by_group(df, group_col='prop_country_id', srch_col='srch_id',
                          score_col='predicted_score', rel_col='relevance', k=5):
    results = []
    for group_val, group_df in df.groupby(group_col):
        ndcg_vals = []
        for _, search_df in group_df.groupby(srch_col):
            if len(search_df) < 2 or search_df[rel_col].sum() == 0:
                continue
            y_true = search_df[rel_col].values.reshape(1, -1)
            y_pred = search_df[score_col].values.reshape(1, -1)
            ndcg = ndcg_score(y_true, y_pred, k=k)
            ndcg_vals.append(ndcg)
        if ndcg_vals:
            results.append({
                group_col: group_val,
                f'NDCG@{k}': np.mean(ndcg_vals),
                'search_count': len(ndcg_vals)
            })
    result_df = pd.DataFrame(results).sort_values(f'NDCG@{k}', ascending=False)
    ndcg_values = result_df[f'NDCG@{k}'].values
    if len(ndcg_values) > 1:
        mean_ndcg = np.mean(ndcg_values)
        std_ndcg = np.std(ndcg_values)
        cv = std_ndcg / mean_ndcg if mean_ndcg != 0 else np.nan
        print(f"Uniformity (CV) of NDCG@{k} across groups: {cv:.4f}")
    else:
        print("Not enough groups to compute uniformity metric.")
    return result_df

def run_param_grid_search(X_train, y_train, group_train, X_val, y_val, group_val, param_grid):
    return train_gbdt_iter(
        X_train, y_train, group_train, X_val, y_val, group_val,
        param_grid, results_csv="gbdt_grid_results_3.csv"
    )