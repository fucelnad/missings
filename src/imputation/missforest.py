import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class MissForest:
    """Class that provides MissForest imputation"""
    # Source:
    # Title: MissForest
    # Code version: 2.4.2
    # Available at: https://github.com/yuenshingyan/MissForest
    # Accessed on: 19-02-2024

    def __init__(self, max_iter: int = 5) -> None:

        self.max_iter = max_iter
        self._mappings = {}
        self._rev_mappings = {}
        self.categorical = None
        self.nan_categorical = None
        self.numerical = None
        self._all_x_imp_cat = []
        self._all_x_imp_num = []
        self._estimators = {}
        self._train_df = None
        self._train_miss = {}
        self._train_obs = {}

    @staticmethod
    def _get_missing_rows(x: pd.DataFrame) -> dict:
        """
        Returns dict that contains features that have missing values as keys and their corresponding indexes as values.
        """

        res = {}
        for c in x.columns:
            feature = x[c]
            is_missing = feature.isnull() > 0
            missing_index = feature[is_missing].index
            if len(missing_index) > 0:
                res[c] = missing_index

        return res

    @staticmethod
    def _get_obs_rows(x: pd.DataFrame) -> dict:
        """
        Returns dict that contains features that have observed values as keys and their corresponding indexes as values.
        """

        res = {}
        for col in x.columns:
            if not x[col].isnull().all():
                res[col] = x.index[~x[col].isna()].tolist()

        return res

    @staticmethod
    def is_column_numeric(col: pd.Series) -> bool:
        """Checks if categorical column is numerical"""

        original_series = col.dropna()
        numeric_series = pd.to_numeric(original_series, errors='coerce')

        if numeric_series.isnull().any():
            return False
        else:
            return True

    def _set_encoding(self, x: pd.DataFrame, categorical: list) -> None:
        """Sets the encoding and decoding of categorical variables."""

        for c in x.columns:
            if c in categorical:
                if not self.is_column_numeric(x[c]):
                    unique = x[c].dropna().unique()
                    n_unique = range(x[c].dropna().nunique())
                    self._mappings[c] = dict(zip(unique, n_unique))
                    self._rev_mappings[c] = dict(zip(n_unique, unique))

    @staticmethod
    def _check_if_all_single_type(x: pd.DataFrame) -> None:
        """Checks if all values in features belong to the same data type."""

        vectorized_type = np.vectorize(type)
        for c in x.columns:
            feature_no_na = x[c].dropna()
            all_type = vectorized_type(feature_no_na)
            all_unique_type = pd.unique(all_type)
            n_type = len(all_unique_type)
            if n_type > 1:
                raise ValueError(f"Feature {c} has more than one datatype.")

    @staticmethod
    def _initial_imputation(x: pd.DataFrame, categorical: list) -> pd.DataFrame:
        """Does the initial imputation. If feature is numerical imputes with meaan, otherwise with mode."""

        for c in x.columns:
            if c in categorical:
                _initial = x[c].mode().values[0]
            else:
                _initial = x[c].mean()
            x[c].fillna(_initial, inplace=True)

        return x

    def _label_encoding(self, x: pd.DataFrame) -> pd.DataFrame:
        """Encodes features."""

        for c in self._mappings:
            x[c] = x[c].map(self._mappings[c])

        return x

    def _rev_label_encoding(self, x: pd.DataFrame) -> pd.DataFrame:
        """Decodes features."""

        for c in self._rev_mappings:
            x[c] = x[c].map(self._rev_mappings[c])

        return x

    def _add_unseen_categories(self, x: pd.DataFrame, mappings: dict) -> [dict, dict]:
        """Adds new categories to mappings."""

        for k, v in mappings.items():
            for category in x[k].unique():
                if category not in v and not pd.isna(category):
                    mappings[k][category] = max(v.values()) + 1

        rev_mappings = {k: {v2: k2 for k2, v2 in v.items()} for k, v in mappings.items()}
        self._mappings, self._rev_mappings = mappings, rev_mappings

    def fit(self, x: pd.DataFrame, categorical: list = None) -> None:
        """Fits estimators on given dataset."""

        # no feature can have all values missing
        if np.any(x.isnull().sum() == len(x)):
            raise ValueError("One or more columns have all rows missing.")

        if categorical is None:
            categorical = []

        x = x.copy()
        self.categorical = categorical
        self.nan_categorical = x[self.categorical].isnull().sum().sum()
        self.numerical = [c for c in x.columns if c not in categorical]
        self._check_if_all_single_type(x)
        self._set_encoding(x, categorical)
        self._set_estimators(x)

    def _set_estimators(self, x: pd.DataFrame) -> None:
        """Sets estimators."""

        self._train_miss = self._get_missing_rows(x)
        self._train_obs = self._get_obs_rows(x)
        x_imp = self._initial_imputation(x, self.categorical)
        x_imp = self._label_encoding(x_imp)

        all_gamma_cat = []
        all_gamma_num = []
        n_iter = 0
        while True:
            x_imp = self._update_imputed_df(x_imp)
            x_imp, all_gamma_cat, all_gamma_num = self._count_convergence(x_imp, all_gamma_cat, all_gamma_num)
            n_iter += 1
            if n_iter > self.max_iter:
                self._train_df = x_imp.copy()
                break
            if self._should_break(n_iter, all_gamma_cat, all_gamma_num):
                self._train_df = x_imp.copy()
                break

    def _update_imputed_df(self, x_imp: pd.DataFrame) -> pd.DataFrame:
        """Imputes missing values based on newly trained estimator."""

        for c in self._train_miss:
            estimator = self._fit_estimator(c, x_imp)
            updated = self._update_missing_values(x_imp, c, estimator)[c]
            self._estimators[c] = estimator
            x_imp[c] = updated

        self._all_x_imp_cat.append(x_imp[self.categorical].reset_index(drop=True))
        self._all_x_imp_num.append(x_imp[self.numerical].reset_index(drop=True))
        return x_imp

    def _fit_estimator(self, c, x_imp: pd.DataFrame):
        """Fits new estimator."""

        if c in self.categorical:
            estimator = RandomForestClassifier(random_state=42)
        else:
            estimator = RandomForestRegressor(random_state=42)

        x_obs = x_imp.drop(c, axis=1).loc[self._train_obs[c]]
        y_obs = x_imp[c].loc[self._train_obs[c]]
        estimator.fit(x_obs, y_obs)

        return estimator

    def _update_missing_values(self, x_imp: pd.DataFrame, c, estimator) -> pd.DataFrame:
        """Updates new estimated values in a dataset."""

        miss_index = self._train_miss[c]
        x_missing = x_imp.loc[miss_index]
        x_missing = x_missing.drop(c, axis=1)
        y_pred = estimator.predict(x_missing)
        y_pred = pd.Series(y_pred)
        y_pred.index = self._train_miss[c]
        x_imp.loc[miss_index, c] = y_pred
        return x_imp

    def _count_convergence(self, x_imp: pd.DataFrame, all_gamma_cat: list, all_gamma_num: list) \
            -> [pd.DataFrame, list, list]:
        """Counts convergence."""

        if (len(self.categorical) > 0 and len(self._all_x_imp_cat) >= 2
                and self.nan_categorical and self.nan_categorical != 0):
            x_imp_cat = self._all_x_imp_cat[-1]
            x_imp_cat_prev = self._all_x_imp_cat[-2]
            gamma_cat = (np.sum(np.sum(x_imp_cat != x_imp_cat_prev, axis=0), axis=0) / self.nan_categorical)
            all_gamma_cat.append(gamma_cat)

        if len(self.numerical) > 0 and len(self._all_x_imp_num) >= 2:
            x_imp_num = self._all_x_imp_num[-1]
            x_imp_num_prev = self._all_x_imp_num[-2]
            gamma_num = (np.sum(np.sum((x_imp_num - x_imp_num_prev) ** 2, axis=0), axis=0) /
                         np.sum(np.sum(x_imp_num ** 2, axis=0), axis=0))
            all_gamma_num.append(gamma_num)

        return x_imp, all_gamma_cat, all_gamma_num

    def _should_break(self, n_iter: int, all_gamma_cat: list, all_gamma_num: list) -> bool:
        """Check if algorithm converges."""

        if (n_iter >= 2 and len(self.categorical) > 0 and len(all_gamma_cat) >= 2 and
                all_gamma_cat[-1] > all_gamma_cat[-2]):
            return True

        if (n_iter >= 2 and len(self.numerical) > 0 and len(all_gamma_cat) >= 2 and
                all_gamma_num[-1] > all_gamma_num[-2]):
            return True

        return False

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Imputes data with fitted estimators."""

        x = x.copy()
        miss = self._get_missing_rows(x)
        self._add_unseen_categories(x, self._mappings)
        x = self._label_encoding(x)

        for c in miss:
            miss_index = miss[c]
            x_missing = x.loc[miss_index]
            x_missing = x_missing.drop(c, axis=1)

            if c in self._estimators:
                y_pred = self._estimators[c].predict(x_missing)
            else:
                estimator = self._fit_estimator(c, self._train_df)
                y_pred = estimator.predict(x_missing)

            # fill in estimated values
            y_pred = pd.Series(y_pred)
            y_pred.index = miss[c]

            if x[c].dtype == "category":
                new_categories = pd.unique(y_pred)
                new_categories = [cat for cat in new_categories if cat not in x[c].cat.categories]
                x[c] = x[c].cat.add_categories(new_categories)
            x.loc[miss[c], c] = y_pred

        x = self._rev_label_encoding(x)
        return x
