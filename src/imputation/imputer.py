import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


def get_categorical_features(df):
    """Returns all features with type category."""

    categorical = [col for col in df.columns if df[col].dtype == "category"]
    if len(categorical) == 0:
        return None

    return categorical


def get_numeric_features(df):
    """Returns all features that can be possibly cast to numeric and those that can not."""

    non_numeric = []  # all cols that can be cast to numerical
    for column in df.columns:
        try:
            df[column].astype(float)
        except ValueError:
            non_numeric.append(column)

    num_cols = [col for col in df.columns if col not in non_numeric]  # all cols that can be cast to numerical

    return num_cols, non_numeric


def one_hot_encode(train, test, cols):
    """Returns train and test dataset with one-hot encoded categorical features respecting NaN values."""

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(train[cols])

    train_encoded = pd.DataFrame(encoder.transform(train[cols]), columns=encoder.get_feature_names_out(cols))
    test_encoded = pd.DataFrame(encoder.transform(test[cols]), columns=encoder.get_feature_names_out(cols))

    train_encoded = train_encoded.reset_index(drop=True)
    test_encoded = test_encoded.reset_index(drop=True)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    train = train.join(train_encoded)
    test = test.join(test_encoded)

    for col in cols:
        nan_mask_train = train[col].isnull()
        nan_mask_test = test[col].isnull()
        train.loc[nan_mask_train, train.columns[train.columns.str.startswith(col + "_")]] = np.nan
        test.loc[nan_mask_test, test.columns[test.columns.str.startswith(col + "_")]] = np.nan

    train.drop(columns=cols, inplace=True)
    test.drop(columns=cols, inplace=True)

    return train, test, encoder

def round_col(cur_df, orig_df, col, unique_values):
    """Rounds numerical column to its nearest value from unique_values."""

    orig_df = orig_df.reset_index(drop=True)
    cur_df = cur_df.reset_index(drop=True)
    mask = orig_df[col].isna()
    cur_df.loc[mask, col] = cur_df.loc[mask, col].apply(lambda x: find_nearest(unique_values, x))
    return cur_df

def find_nearest(array, value):
    """Finds nearest number from array to given value."""

    array = np.asarray(array)
    array = array[~np.isnan(array)]
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def round_cat_cols(train_res, test_res, orig, num_cols):
    """Rounds numerical columns to the closest category"""

    train_orig, test_orig = orig
    for col in train_orig:
        if train_orig[col].dtype == "category" and col in num_cols and col in train_res.columns:
            unique_values = train_orig[col].unique()
            train_res = round_col(train_res, train_orig, col, unique_values)
            test_res = round_col(test_res, test_orig, col, unique_values)
            train_res[col] = train_res[col].round(decimals=2).astype('category')
            test_res[col] = test_res[col].round(decimals=2).astype('category')
    return train_res, test_res


def normalize(train, test):
    """Normalizes the data."""

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    return train_scaled, test_scaled, scaler
