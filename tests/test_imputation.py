import dash
import numpy as np
import pandas as pd
import pytest

from src.imputation import imputer
from src.imputation.gain_imputer import GAINImputation
from src.imputation.knn_imputer import KNNImputation
from src.imputation.mean_imputer import MeanImputation
from src.imputation.mice_imputer import MICEImputation
from src.imputation.missforest_imputer import MissForestImputation


app = dash.Dash(__name__, prevent_initial_callbacks='initial_duplicate')


@pytest.fixture
def mean_imputation():
    instance = MeanImputation(app)
    return instance


@pytest.fixture
def missforest_imputation():
    instance = MissForestImputation(app)
    return instance


@pytest.fixture
def knn_imputation():
    instance = KNNImputation(app)
    return instance


@pytest.fixture
def mice_imputation():
    instance = MICEImputation(app)
    return instance


@pytest.fixture
def gain_imputation():
    instance = GAINImputation(app)
    return instance


def test_mean_imputation(mean_imputation):
    """Tests mean imputation. Numerical as well as non-numerical features.
    Tests if given dataset remains unchanged."""

    df_train = pd.DataFrame({'A': [1, 2, np.nan, 4, 6],
                             'B': ['apple', 1, np.nan, 'apple', 'banana'],
                             'target': [1, 2, 3, 4, 5]})
    df_test = pd.DataFrame({'A': [np.nan, 2, 3, 4, 5],
                            'B': [np.nan, 'apple', 'banana', 'banana', 'banana'],
                            'target': [1, 2, 3, 4, 5]})

    df_train_orig = df_train.copy(deep=True)
    df_test_orig = df_test.copy(deep=True)

    data = (df_train, df_test)
    train_expected = pd.DataFrame({'A': [1.0, 2.0, 3.0, 4.0, 6.0],
                                   'B': ['apple', 1, 'apple', 'apple', 'banana'],
                                  'target': [1, 2, 3, 4, 5]
    })
    test_expected = pd.DataFrame({'A': [3.0, 2.0, 3.0, 4.0, 5.0],
                                  'B': ['apple', 'apple', 'banana', 'banana', 'banana'],
                                  'target': [1, 2, 3, 4, 5]})

    imputed, _ = mean_imputation.impute_data(data, ['A'], 'target')
    train_imputed, test_imputed = imputed

    pd.testing.assert_frame_equal(train_imputed, train_expected)
    pd.testing.assert_frame_equal(test_imputed, test_expected)
    pd.testing.assert_frame_equal(df_train_orig, df_train)
    pd.testing.assert_frame_equal(df_test_orig, df_test)


def test_missforest_imputation(missforest_imputation):
    """Tests MissForest imputation. Numerical as well as non-numerical features.
        Tests if given dataset remains unchanged. Tests if non-missing values stay the same."""

    df_train = pd.DataFrame({'A': [1, 2, np.nan, 4, 5],
                             'B': ['apple', "orange", np.nan, 'apple', 'banana'],
                             'target': [1, 2, 3, 4, 5]})
    df_test = pd.DataFrame({'A': [np.nan, 2, 3, 4, 5],
                            'B': [np.nan, 'apple', 'banana', 'banana', 'banana'],
                            'target': [1, 2, 3, 4, 5]})
    df_train['B'] = df_train['B'].astype('category')
    df_test['B'] = df_test['B'].astype('category')

    df_train_orig = df_train.copy(deep=True)
    df_test_orig = df_test.copy(deep=True)

    data = (df_train, df_test)
    imputed, _ = missforest_imputation.impute_data(data, ['A'], 'target')
    train_imputed, test_imputed = imputed

    assert (np.array_equal(train_imputed["A"][0:2].values, np.array([1, 2])) &
            np.array_equal(train_imputed["A"][3:5].values, np.array([4, 5])))
    assert (np.array_equal(train_imputed["B"][0:2].values, np.array(['apple', "orange"])) &
            np.array_equal(train_imputed["B"][3:5].values, np.array(['apple', 'banana'])))
    assert np.array_equal(test_imputed["A"][1:].values, np.array([2, 3, 4, 5]))
    assert np.array_equal(test_imputed["B"][1:].values, np.array(['apple', 'banana', 'banana', 'banana']))
    pd.testing.assert_frame_equal(df_train_orig, df_train)
    pd.testing.assert_frame_equal(df_test_orig, df_test)

    df_train = pd.DataFrame({'A': [1, 2, 3, 4, 5],
                             'B': ['apple', "orange", "pineapple", 'apple', 'banana'],
                             'target': [1, 2, 3, 4, 5]})
    df_test = pd.DataFrame({'A': [np.nan, 2, 3, 4, 5],
                            'B': [np.nan, 'apple', 'banana', 'banana', 'banana'],
                             'target': [1, 2, 3, 4, 5]})
    df_train['B'] = df_train['B'].astype('category')
    df_test['B'] = df_test['B'].astype('category')

    data = (df_train, df_test)
    imputed, _ = missforest_imputation.impute_data(data, [], 'target')
    train_imputed, test_imputed = imputed

    assert np.array_equal(train_imputed["A"].values, np.array([1, 2, 3, 4, 5]))
    assert np.array_equal(train_imputed["B"].values, np.array(['apple', "orange", "pineapple", 'apple', 'banana']))
    assert np.array_equal(test_imputed["A"][1:].values, np.array([2, 3, 4, 5]))
    assert np.array_equal(test_imputed["B"][1:].values, np.array(['apple', 'banana', 'banana', 'banana']))


def test_knn_imputation(knn_imputation):
    """Tests kNN imputation"""

    df_train = pd.DataFrame({'A': [600, 20, np.nan, 400, 56],
                             'B': ['apple', "orange", np.nan, 'apple', 'banana'],
                             'target': [1, 2, 3, 4, 5]})
    df_test = pd.DataFrame({'A': [np.nan, 2, 3, 4, 5],
                            'B': [np.nan, 'apple', 'banana', 'coconut', 'banana'],
                             'target': [1, 2, 3, 4, 5]})
    df_train['B'] = df_train['B'].astype('category')
    df_test['B'] = df_test['B'].astype('category')

    df_train_orig = df_train.copy(deep=True)
    df_test_orig = df_test.copy(deep=True)

    data = (df_train, df_test)
    imputed, _ = knn_imputation.impute_data(data, [], 'target')
    train_imputed, test_imputed = imputed

    train_expected = pd.DataFrame({'B': ['apple', "orange", "apple", 'apple', 'banana'],
                                   'A': [600.0, 20.0, 269.0, 400.0, 56.0],
                             'target': [1, 2, 3, 4, 5]})
    test_expected = pd.DataFrame({'B': ['apple', 'apple', 'banana', 'coconut', 'banana'],
                                  'A': [269.0, 2.0, 3.0, 4.0, 5.0],
                             'target': [1, 2, 3, 4, 5]})
    train_expected['B'] = train_expected['B'].astype('category')
    test_expected['B'] = test_expected['B'].astype('category')

    pd.testing.assert_frame_equal(train_imputed, train_expected)
    pd.testing.assert_frame_equal(test_imputed, test_expected)
    pd.testing.assert_frame_equal(df_train_orig, df_train)
    pd.testing.assert_frame_equal(df_test_orig, df_test)

    df_train['A'] = df_train['A'].astype('category')
    df_test['A'] = df_test['A'].astype('category')

    data = (df_train, df_test)
    imputed, _ = knn_imputation.impute_data(data, [], 'target')
    train_imputed, test_imputed = imputed

    train_expected = pd.DataFrame({'B': ['apple', "orange", "apple", 'apple', 'banana'],
                                   'A': [600.0, 20.0, 400.0, 400.0, 56.0],
                                   'target': [1, 2, 3, 4, 5]})
    test_expected = pd.DataFrame({'B': ['apple', 'apple', 'banana', 'coconut', 'banana'],
                                  'A': [400.0, 2.0, 3.0, 4.0, 5.0],
                                  'target': [1, 2, 3, 4, 5]})
    train_expected['B'] = train_expected['B'].astype('category')
    test_expected['B'] = test_expected['B'].astype('category')
    train_expected['A'] = train_expected['A'].astype('category')
    test_expected['A'] = test_expected['A'].astype('category')

    train_imputed['A'] = train_imputed['A'].astype(float)
    train_imputed['A'] = train_imputed['A'].round(decimals=2)
    train_imputed['A'] = train_imputed['A'].astype('category')

    pd.testing.assert_frame_equal(train_imputed, train_expected)
    pd.testing.assert_frame_equal(test_imputed, test_expected)


def test_mice_imputation(mice_imputation):
    """Tests MICE imputation."""

    df_train = pd.DataFrame({'A': [600, 20, np.nan, 400, 56],
                             'B': ['apple', "orange", np.nan, 'apple', 'banana'],
                             'target': [1, 2, 3, 4, 5]})
    df_test = pd.DataFrame({'A': [np.nan, 2, 3, 4, 5],
                            'B': [np.nan, 'apple', 'banana', 'coconut', 'banana'],
                            'target': [1, 2, 3, 4, 5]})

    df_train['A'] = df_train['A'].astype('category')
    df_test['A'] = df_test['A'].astype('category')
    df_train['B'] = df_train['B'].astype('category')
    df_test['B'] = df_test['B'].astype('category')

    df_train_orig = df_train.copy(deep=True)
    df_test_orig = df_test.copy(deep=True)

    data = (df_train, df_test)
    imputed, _ = mice_imputation.impute_data(data, [], 'target')
    train_imputed, test_imputed = imputed

    assert train_imputed['B'].equals(pd.Series(['apple', "orange", "apple", 'apple', 'banana']).astype('category'))
    assert test_imputed['B'].equals(pd.Series(['apple', 'apple', 'banana', 'coconut', 'banana']).astype('category'))
    pd.testing.assert_frame_equal(df_train_orig, df_train)
    pd.testing.assert_frame_equal(df_test_orig, df_test)


def test_one_hot_encoding():
    """Tests one-hot encoding."""

    train = pd.DataFrame({
        'A': ['a', 'b', 'a', np.nan],
        'B': ['b', 'a', np.nan, 'a'],
        'C': [1, 2, 3, 4]
    })
    test = pd.DataFrame({
        'A': ['b', np.nan],
        'B': ['b', 'b'],
        'C': [5, 6]
    })

    train_expected = pd.DataFrame({
        'C': [1, 2, 3, 4],
        'A_a': [1.0, 0.0, 1.0, np.nan],
        'A_b': [0.0, 1.0, 0.0, np.nan],
        'A_nan': [0.0, 0.0, 0.0, np.nan],
        'B_a': [0.0, 1.0, np.nan, 1.0],
        'B_b': [1.0, 0.0, np.nan, 0.0],
        'B_nan': [0.0, 0.0, np.nan, 0.0],
    })

    test_expected = pd.DataFrame({
        'C': [5, 6],
        'A_a': [0.0, np.nan],
        'A_b': [1.0, np.nan],
        'A_nan': [0.0, np.nan],
        'B_a': [0.0, 0.0],
        'B_b': [1.0, 1.0],
        'B_nan': [0.0, 0.0]
    })

    train_encoded, test_encoded, _ = imputer.one_hot_encode(train, test, ["A", "B"])
    pd.testing.assert_frame_equal(train_encoded, train_expected)
    pd.testing.assert_frame_equal(test_encoded, test_expected)


def test_find_nearest():
    """Tests finding nearest value to number from given array."""

    unique_values = [1, 2, 3, 4.5]
    assert imputer.find_nearest(unique_values, 2.5) == 2
    assert imputer.find_nearest(unique_values, 3.6) == 3
    assert imputer.find_nearest(unique_values, 4) == 4.5


def test_round_col():
    """Tests rounding categorical numerical column."""

    orig = pd.DataFrame({
        'col_num': [1.1, 2.2, 3.3, np.nan],
        'col_cat': ['a', 'b', 'c', 'd']
    })

    imp = pd.DataFrame({
        'col_num': [1.1, 2.2, 3.3, 4.4],
        'col_cat': ['a', 'b', 'c', 'd']
    })
    unique_values = [1, 2, 3, 4]

    rounded_df = imputer.round_col(imp, orig, 'col_num', unique_values)
    assert (rounded_df['col_num'] == [1.1, 2.2, 3.3, 4]).all()


def test_get_nearest_cat():
    """Tests finding nearest categories in GAIN."""

    imputer = GAINImputation(app)
    train_orig = pd.DataFrame({
        'num_col': pd.Series([3, 4, 5, np.nan], dtype='category'),
        'other_col': pd.Series([1, 2, 3])
    })

    imputer.train = pd.DataFrame({
        'num_col': pd.Series([3, 4, 5, 5.5], dtype='category'),
        'other_col': pd.Series([1, 2, 3, 4])
    })

    imputer.test = pd.DataFrame({
        'num_col': pd.Series([3, 4, 5, 3.3], dtype='category'),
        'other_col': pd.Series([1, 2, 3, 4])
    })
    imputer.num_cols = ['num_col', 'other_col']

    mask_train = train_orig[imputer.num_cols].apply(pd.isnull)
    imputer.train, imputer.test = imputer.get_nearest_cat(train_orig, mask_train, mask_train)

    expected_series = pd.Series([3.0, 4.0, 5.0, 5.0]).astype("category")
    expected_series = expected_series.cat.set_categories([3.0, 4.0, 5.0])
    assert imputer.train['num_col'].equals(expected_series)

    expected_series = pd.Series([3.0, 4.0, 5.0, 3.0]).astype("category")
    expected_series = expected_series.cat.set_categories([3.0, 4.0, 5.0])
    assert imputer.test['num_col'].equals(expected_series)


def test_gain_imputation(gain_imputation):
    """Tests GAIN imputation."""

    df_train = pd.DataFrame({'A': [600, 20, np.nan, 400, 56],
                             'B': ['apple', "orange", np.nan, 'apple', 'banana'],
                             'target': [1, 2, 3, 4, 5]})
    df_test = pd.DataFrame({'A': [np.nan, 2, 3, 4, 5],
                            'B': [np.nan, 'apple', 'banana', 'coconut', 'banana'],
                            'target': [1, 2, 3, 4, 5]})

    df_train['A'] = df_train['A'].astype('category')
    df_test['A'] = df_test['A'].astype('category')
    df_train['B'] = df_train['B'].astype('category')
    df_test['B'] = df_test['B'].astype('category')

    df_train_orig = df_train.copy(deep=True)
    df_test_orig = df_test.copy(deep=True)

    data = (df_train, df_test)
    gain_imputation.params = {"batch_size": 1, "hint_rate": 0.9, "alpha": 100, "iter": 200}
    imputed, _ = gain_imputation.impute_data(data, [], 'target')
    train_imputed, test_imputed = imputed

    assert train_imputed['B'].equals(pd.Series(['apple', "orange", "apple", 'apple', 'banana']).astype('category'))
    assert test_imputed['B'].equals(pd.Series(['apple', 'apple', 'banana', 'coconut', 'banana']).astype('category'))
    pd.testing.assert_frame_equal(df_train_orig, df_train)
    pd.testing.assert_frame_equal(df_test_orig, df_test)
