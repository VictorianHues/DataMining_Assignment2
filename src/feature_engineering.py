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