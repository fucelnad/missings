import numpy as np
import pandas as pd
import pytest
from src.ml_training.trainer import impute_data

def test_impute_data():
    train = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': ['a', 'b', 'c', np.nan],
        'C': [np.nan, np.nan, np.nan, np.nan]
    })
    train['B'] = train['B'].astype('category')

    test = pd.DataFrame({
        'A': [5, np.nan, 7, 8],
        'B': ['d', np.nan, 'f', 'g'],
        'C': [np.nan, np.nan, np.nan, 10]
    })
    test['B'] = test['B'].astype('category')

    train_imputed, test_imputed = impute_data(train, test)

    assert not train_imputed.isnull().any().any()
    assert not test_imputed.isnull().any().any()

    assert 'A_was_missing' in train_imputed.columns
    assert 'B_was_missing' in train_imputed.columns
    assert 'C_was_missing' in train_imputed.columns
    assert 'A_was_missing' in test_imputed.columns
    assert 'B_was_missing' in test_imputed.columns
    assert 'C_was_missing' in test_imputed.columns

    assert "-1" in train_imputed['B'].cat.categories
    assert "-1" in test_imputed['B'].cat.categories

    assert (train_imputed.loc[train['A'].isnull(), 'A'] == -1).all()
    assert (test_imputed.loc[test['A'].isnull(), 'A'] == -1).all()
    assert (train_imputed.loc[train['B'].isnull(), 'B'] == "-1").all()
    assert (test_imputed.loc[test['B'].isnull(), 'B'] == "-1").all()
    assert (train_imputed.loc[train['C'].isnull(), 'C'] == -1).all()
    assert (test_imputed.loc[test['C'].isnull(), 'C'] == -1).all()
