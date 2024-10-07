import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.imputation.missforest import MissForest


@pytest.fixture
def setup():
    return MissForest()


def test_get_missing_rows(setup):
    obj = setup
    df = pd.DataFrame({
        'A': [1, 2, np.nan],
        'B': [4, np.nan, 6]
    })
    result = obj._get_missing_rows(df)
    assert result == {'A': pd.Index([2]), 'B': pd.Index([1])}


def test_get_obs_rows(setup):
    df = pd.DataFrame({
        'A': [1, 2, np.nan],
        'B': [np.nan, np.nan, np.nan],
        'C': [3, 4, 5]
    })

    obj = setup
    result = obj._get_obs_rows(df)
    assert result == {'A': [0, 1], 'C': [0, 1, 2]}

def test_set_encoding(setup):
    df = pd.DataFrame({
        'A': ['a', 'b', 'a'],
        'B': [1, 2, 3]
    })

    expected_mappings = {'A': {'a': 0, 'b': 1}}
    expected_rev_mappings = {'A': {0: 'a', 1: 'b'}}
    obj = setup

    obj._set_encoding(df, ["A"])
    assert obj._mappings == expected_mappings
    assert obj._rev_mappings == expected_rev_mappings


def test_check_if_all_single_type(setup):
    df = pd.DataFrame({
        'A': [1, 2, '3'],
        'B': [4.0, 5.0, 6.0]
    })
    obj = setup

    with pytest.raises(ValueError):
        obj._check_if_all_single_type(df)


def test_initial_imputation(setup):
    df = pd.DataFrame({
        'A': [1, 2, np.nan],
        'B': ['a', np.nan, 'b']
    })
    expected_result = pd.DataFrame({
        'A': [1, 2, 1.5],
        'B': ['a', 'a', 'b']
    })
    obj = setup

    result = obj._initial_imputation(df, ["B"])
    pd.testing.assert_frame_equal(result, expected_result)


def test_label_encoding(setup):
    df = pd.DataFrame({
        'A': ['a', 'b', 'a']
    })
    obj = setup
    obj._mappings = {'A': {'a': 0, 'b': 1}}
    expected_result = pd.DataFrame({
        'A': [0, 1, 0]
    }, dtype=np.int64)

    result = obj._label_encoding(df)
    pd.testing.assert_frame_equal(result, expected_result)


def test_rev_label_encoding(setup):
    df = pd.DataFrame({
        'A': [0, 1, 0]
    })
    obj = setup
    obj._rev_mappings = {'A': {0: 'a', 1: 'b'}}
    expected_result = pd.DataFrame({
        'A': ['a', 'b', 'a']
    })

    result = obj._rev_label_encoding(df)
    pd.testing.assert_frame_equal(result, expected_result)


def test_add_unseen_categories(setup):
    obj = setup
    x = pd.DataFrame({
        'col1': ['a', 'b', 'c'],
        'col2': ['x', 'y', 'z']
    })
    mappings = {'col1': {'a': 1, 'b': 2}, 'col2': {'x': 1, 'y': 2}}

    obj._add_unseen_categories(x, mappings)
    expected_mappings = {'col1': {'a': 1, 'b': 2, 'c': 3}, 'col2': {'x': 1, 'y': 2, 'z': 3}}
    expected_rev_mappings = {'col1': {1: 'a', 2: 'b', 3: 'c'}, 'col2': {1: 'x', 2: 'y', 3: 'z'}}

    assert obj._mappings == expected_mappings
    assert obj._rev_mappings == expected_rev_mappings


def test_fit(setup):
    obj = setup
    x = pd.DataFrame({
        'col1': ['a', 'b', 'c'],
        'col2': [1, 2, 3]
    })

    with pytest.raises(ValueError):
        obj.fit(x.assign(col1=None), ['col1'])

    obj.fit(x, ['col1'])
    assert obj.categorical == ['col1']
    assert obj.numerical == ['col2']


def test_fit_estimator(setup):
    obj = setup
    x_imp = pd.DataFrame({
        'col1': [1, 2, np.nan],
        'col2': [1, np.nan, 2]
    })
    obj.categorical = ['col1']
    obj._train_obs = {'col1': x_imp['col1'].notna(), 'col2': x_imp['col2'].notna()}

    estimator = obj._fit_estimator('col1', x_imp)
    assert isinstance(estimator, RandomForestClassifier)

    estimator = obj._fit_estimator('col2', x_imp)
    assert isinstance(estimator, RandomForestRegressor)
