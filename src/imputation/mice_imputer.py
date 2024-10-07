import base64
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import urllib

import src.imputation.imputer as imputer
from src.imputation.template_imputer import Imputer


class MICEImputation(Imputer):

    def __init__(self, app):
        super().__init__(app)
        self.name = "MICE"
        self.params = {"num_datasets": 3, "iter": 10}

    def provide_info(self):

        encoded_image = base64.b64encode(open("src/program_files/mice.png", "rb").read())
        encoded_image = "data:image/png;base64," + urllib.parse.quote(encoded_image)
        image = html.Img(src=encoded_image, style={"height": "60%", "width": "60%", "display": "block",
                                                   "margin-left": "auto", "margin-right": "auto"})
        text = html.Div([
            html.P("""
                Multivariate imputation by chained equations (MICE) is an iterative statistical imputation method. 
                It consists of the following steps:"""),
            html.Ol([
                html.Li("Remembering where values are missing and performing mean imputation for each missing value."),
                html.Li(
                    "A feature is selected and set back to original – imputed values in this feature "
                    "are set back to missing."),
                html.Li(
                    "Observed values in the selected feature are used to train the regression model. In this "
                    "application Bayesian Ridge Regression is used. Values are regressed on the rest of the features "
                    "in the dataset."),
                html.Li(
                    "The missing values in the selected feature are replaced by predictions from the regression model.")
            ]),
            html.P(
                """
                Steps 2–4 are repeated for each feature with missing data, representing an iteration. The order of the 
                features in this case is ascending, from those with the fewest missing values to those with the most. 
                There may be more iterations, whose number may be adjusted above as a hypeparameter. It defines the number of iterations to perform 
                before returning the final imputations."""),
            html.P("""
                Once the desired number of iterations has been completed, the whole process is repeated from the 
                beginning, resulting in another complete dataset. These datasets differ only in the places where data values
                were missing. The datasets are then merged into a final dataset where the mean of the imputed values is
                 calculated."""),
            html.P(["""
                Features with float, int and category datatypes that can be cast to numeric datatypes are imputed
                 as described. However, int data types are rounded to the nearest integer at the end. Numeric features 
                 set as category are rounded to the nearest category. Features with category data types that can not be 
                 cast to numeric data types are imputed using mode and one-hot encoded for the purpose of training 
                 regression models. [""",
                   html.A("1", href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3074241/", target="_blank"),
                    ", ",
                    html.A("2", target="_blank",
                           href="https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html"),
                    "]"]),
            image,
            html.P(["MICE imputation.  Figure adapted from [", html.A("3", target="_blank",
                                                 href="https://bookdown.org/mwheymans/bookmi/multiple-imputation.html"),
                    "]."], style={"textAlign": "center"})
        ], style={"margin-left": "100px", "margin-right": "100px", "textAlign": "justify"})
        return text

    def param_tuning(self):
        """Options for number of datasets created during imputation"""

        return html.Div([
            html.H5("Select number of datasets created during imputation:"),
            dcc.Slider(
                id='mice-slider',
                min=1, max=7,
                step=1, value=3
            ),
            html.H5("Select number of iterations:"),
            dcc.Slider(
                id='mice-iter',
                min=5, max=20,
                step=1, value=10
            ),
        ], style={"margin-left": "100px", "margin-right": "100px"})

    def get_nearest_categories(self, mean_train, mean_test, mask_train, mask_test):
        """Finds nearest categories to imputed means in case the feature was categorical."""

        for col in self.train:
            if self.train[col].dtype == "category" and col in self.num_cols:
                unique_values = self.train[col].unique()
                mean_train.loc[mask_train[col], col] = mean_train.loc[mask_train[col], col].apply(
                    lambda x: imputer.find_nearest(unique_values, x))
                mean_test.loc[mask_test[col], col] = mean_test.loc[mask_test[col], col].apply(
                    lambda x: imputer.find_nearest(unique_values, x))

        return mean_train, mean_test

    def mice_imputation(self):
        """Imputes data using MICE."""

        mask_train = self.train[self.num_cols].apply(pd.isnull)
        mask_test = self.test[self.num_cols].apply(pd.isnull)
        mean_train = pd.DataFrame(0, index=self.train.index, columns=self.train.columns)
        mean_test = pd.DataFrame(0, index=self.test.index, columns=self.test.columns)

        for i in range(self.params["num_datasets"]):
            method = IterativeImputer(sample_posterior=True, random_state=i, max_iter=self.params["iter"])
            train_imp = pd.DataFrame(method.fit_transform(self.train), columns=self.train.columns)
            test_imp = pd.DataFrame(method.transform(self.test), columns=self.test.columns)

            mean_train[mask_train] += train_imp[mask_train]
            mean_test[mask_test] += test_imp[mask_test]

        mean_train[mask_train] /= self.params["num_datasets"]
        mean_test[mask_test] /= self.params["num_datasets"]
        mean_train, mean_test = self.get_nearest_categories(mean_train, mean_test, mask_train, mask_test)

        self.train[self.num_cols] = self.train[self.num_cols].astype('float')
        self.test[self.num_cols] = self.test[self.num_cols].astype('float')
        self.train.fillna(mean_train, inplace=True)
        self.test.fillna(mean_test, inplace=True)

    def impute_data(self, data, int64_cols, target):
        """Handles MICE imputation."""

        if not data[0].isna().any().any() and not data[1].isna().any().any():
            return (data[0], data[1]), None

        self.train, self.test = data[0].copy(), data[1].copy()
        # drop target before imputation
        train_target = self.train[target]
        test_target = self.test[target]
        self.train.drop(target, axis=1, inplace=True)
        self.test.drop(target, axis=1, inplace=True)

        self.num_cols, self.non_num_cols = imputer.get_numeric_features(self.train)
        encoder = self.handle_categorical()
        self.mice_imputation()

        if encoder:
            self.decode_categorical(encoder, data[1])

        self.train, self.test = self.original_types((self.train, self.test), data)
        self.train, self.test = self.round_int(self.train, self.test, int64_cols)

        self.train[target] = train_target
        self.test[target] = test_target

        return (self.train, self.test), None

    def params_callback(self):
        """Sets number of neighbours"""

        @self.app.callback(
            Output("imputation-params", "data", allow_duplicate=True),
            Input('mice-slider', 'value'),
            Input("mice-iter", "value"),
            prevent_initial_call=True
        )
        def set_val(num_sets, iter):
            self.params["num_datasets"] = num_sets
            self.params["iter"] = iter
            return dash.no_update

