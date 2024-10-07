from dash import html
import pandas as pd

from src.imputation.template_imputer import Imputer


class MeanImputation(Imputer):

    def __init__(self, app):
        super().__init__(app)
        self.name = "Mean"
        self.params = {}

    def provide_info(self):

        text = html.P("The mean imputation calculates the mean for each missing feature with either float or int "
                      "data type based on the training set. For categorical features, the mode is calculated based on the "
                      "training set. These values are then imputed to both the training and test sets.",
                      style={"margin-left": "100px", "margin-right": "100px", "textAlign": "justify"})
        return text

    def param_tuning(self):
        return None

    def impute_data(self, data, int64_cols, _):
        """Imputes data with mean if numerical and mode otherwise."""

        if not data[0].isna().any().any() and not data[1].isna().any().any():
            return (data[0], data[1]), None

        train, test = data[0].copy(), data[1].copy()

        for col in train.columns:
            if train[col].dtype == "float64":
                train_mean = train[col].mean()
                train[col].fillna(train_mean, inplace=True)
                test[col].fillna(train_mean, inplace=True)
            else:
                train_mode = train[col].mode()[0]
                train[col].fillna(train_mode, inplace=True)
                test[col].fillna(train_mode, inplace=True)

        train, test = self.round_int(train, test, int64_cols)
        return (train, test), None

    def params_callback(self):
        pass
