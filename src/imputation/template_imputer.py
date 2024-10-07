from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

import src.imputation.imputer as imputer


class Imputer(ABC):
    """A base class for imputers"""

    def __init__(self, app):
        self.app = app
        self.non_num_cols = None
        self.num_cols = None
        self.rows = None
        self.name = None  # shown in the summary graph

        self.train = None
        self.test = None
        self.params = {}

    @abstractmethod
    def provide_info(self):
        """Returns theoretical information about imputation method"""
        pass

    @abstractmethod
    def param_tuning(self):
        """Returns options for hyperparam tuning (e.g. sliders, dropdowns, ...)"""
        pass

    @abstractmethod
    def impute_data(self, data, int64_cols, target):
        """Returns imputed data and modal in following format: (train set, test set), pop-up window.
        Pop-up window may be used to inform the user if something did not go as expected during imputation.
        In case you do not want to inform the user set pop-up window to None."""
        pass

    @abstractmethod
    def params_callback(self):
        """Callback for hyperparam tuning. You may set self.params according to selected values."""
        pass

    def handle_categorical(self):
        """Imputes median in categorical columns and one-hot encodes them."""

        # impute median for categorical columns
        for col in self.non_num_cols:
            imputer = SimpleImputer(strategy='most_frequent')
            self.train[col] = imputer.fit_transform(self.train[col].values.reshape(-1, 1))[:, 0]
            self.test[col] = imputer.transform(self.test[col].values.reshape(-1, 1))[:, 0]

        if not self.non_num_cols:
            return None

        # one-hot encode categorical cols
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoder.fit(self.train[self.non_num_cols])
        train_encoded = pd.DataFrame(encoder.transform(self.train[self.non_num_cols]),
                                     columns=encoder.get_feature_names_out(self.non_num_cols))
        test_encoded = pd.DataFrame(encoder.transform(self.test[self.non_num_cols]),
                                    columns=encoder.get_feature_names_out(self.non_num_cols))

        self.train = pd.concat([self.train[self.num_cols].reset_index(drop=True),
                                train_encoded.reset_index(drop=True)], axis=1)
        self.test = pd.concat([self.test[self.num_cols].reset_index(drop=True),
                               test_encoded.reset_index(drop=True)], axis=1)

        return encoder

    def decode_categorical(self, encoder, test_orig):
        """Decodes one-hot encoded columns back to original."""

        # decode one-hot encoded categories back
        col_to_enc = encoder.get_feature_names_out(self.non_num_cols)
        cat_train = pd.DataFrame(encoder.inverse_transform(self.train[col_to_enc]), columns=self.non_num_cols)
        cat_test = pd.DataFrame(encoder.inverse_transform(self.test[col_to_enc]), columns=self.non_num_cols)

        # put numerical and categorical cols to one dataframe
        self.train = pd.concat([self.train[self.num_cols].reset_index(drop=True),
                                cat_train.reset_index(drop=True)], axis=1)
        self.test = pd.concat([self.test[self.num_cols].reset_index(drop=True),
                               cat_test.reset_index(drop=True)], axis=1)
        self.test.fillna(test_orig, inplace=True)

    def original_types(self, curr, orig):
        """Puts imputed features back to their original data type."""

        train, test = curr
        for col in train.columns:
            if orig[0][col].dtype == "category":
                train[col] = train[col].astype("category")
                test[col] = test[col].astype("category")
            else:
                train[col] = train[col].astype(orig[0][col].dtype)
                test[col] = test[col].astype(orig[0][col].dtype)

        return train, test

    def round_int(self, train, test, int64_cols):
        """Rounds imputed integer columns."""

        for col in int64_cols:
            if col in train.columns and train[col].dtype == 'float64':
                train[col] = train[col].apply(np.round)
                test[col] = test[col].apply(np.round)
        return train, test
