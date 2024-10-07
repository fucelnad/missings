import base64
import dash
import pandas as pd
from dash import html, dcc
from dash.dependencies import Input, Output
from sklearn.impute import KNNImputer
import urllib

import src.imputation.imputer as imputer
from src.imputation.template_imputer import Imputer


class KNNImputation(Imputer):

    def __init__(self, app):
        super().__init__(app)
        self.name = "kNN"
        self.params = {"neighbors": 3}
        self.encoder = None
        self.scaler = None

    def provide_info(self):

        encoded_image = base64.b64encode(open("src/program_files/knn.png", "rb").read())
        encoded_image = "data:image/png;base64," + urllib.parse.quote(encoded_image)
        image = html.Img(src=encoded_image, style={"height": "40%", "width": "40%", "display": "block",
                                                   "margin-left": "auto", "margin-right": "auto"})
        text = html.Div([
            html.P("""
                K-Nearest Neighbours (kNN) is an imputation technique that estimates the values based on the 
                neighbouring data points. kNN assumes that similar data points have similar values. Before imputation 
                all features present in the data set are normalised."""),
            html.Ol([
                html.Li("A missing value from a record in the dataset is selected."),
                html.Li("The Euclidean distance between the record and all the other records is calculated based on "
                        "observed features."),
                html.Li("K records with the lowest Euclidean distance are selected. The number of records can be "
                        "adjusted by selecting neighbours. Note that two samples are close if the features that neither"
                        " is missing are close."),
                html.Li(
                    "For the values of these records that are in the same feature as the value to be imputed, "
                    "mean is calculated and used as the imputation value.")
            ]),
            html.P("""Steps 1–4 are repeated for each missing value."""),
            html.P(["""
                    Features with float, int and category data types that can be cast to numeric data types are imputed
                     as described. However, int data types are rounded to the nearest integer at the end. Numeric 
                     features set as category are rounded to the nearest category. Features with category data types that 
                      can not be cast to numeric data types are one-hot encoded and the category with highest imputed 
                      number is used for filling missing value. [""",
                    html.A("1", href="https://dl.acm.org/doi/pdf/10.1145/3411408.3411465", target="_blank"),
                    ", ",
                    html.A("2", target="_blank",
                           href="https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html"),
                    "]"]),
            image,
            html.P(["kNN imputation. Figure adapted from [",
                    html.A("3", target="_blank",
                           href="https://www.researchgate.net/publication/312507655_Learning_k_for_kNN_Classification"),
                    "]."], style={"textAlign": "center"})
        ], style={"margin-left": "100px", "margin-right": "100px", "textAlign": "justify"})
        return text

    def param_tuning(self):
        """Options for number of neighbors"""

        return html.Div([
            html.H5("Select number of neighbours:"),
            dcc.Slider(
                id='knn-slider',
                min=1, max=7,
                step=1, value=3,
            ),
        ], style={"margin-left": "100px", "margin-right": "100px"})

    def process_imp_data(self, df, all_cols, to_encode, numeric, orig):

        df = pd.DataFrame(df, columns=all_cols)
        df = self.scaler.inverse_transform(df)
        df = pd.DataFrame(df, columns=all_cols)

        num = df[numeric]

        if self.encoder:
            encoded_columns = self.encoder.get_feature_names_out(to_encode)
            cat = df[encoded_columns].values
            cat = self.encoder.inverse_transform(cat)
            cat = pd.DataFrame(cat, columns=to_encode)

            # due to label encoding new categories in test set are set as None
            # replacing None with original values
            cat.fillna(orig, inplace=True)
            return cat.join(num)

        return num

    def impute_data(self, data, int64_cols, target):
        """Imputes data with kNN imputation."""

        if not data[0].isna().any().any() and not data[1].isna().any().any():
            return (data[0], data[1]), None

        train, test = data[0].copy(), data[1].copy()
        # drop target before imputation
        train_target = train[target]
        test_target = test[target]
        train.drop(target, axis=1, inplace=True)
        test.drop(target, axis=1, inplace=True)

        num_cols, to_encode = imputer.get_numeric_features(train)
        self.encoder, self.scaler = None, None

        if to_encode:
            # one-hot encoding of non-numeric cols
            train, test, self.encoder = imputer.one_hot_encode(train, test, to_encode)

        all_cols = train.columns
        train, test, self.scaler = imputer.normalize(train, test)  # normalize the data
        knn_imputer = KNNImputer(n_neighbors=self.params["neighbors"])  # impute the data
        train_imp = knn_imputer.fit_transform(train)
        test_imp = knn_imputer.transform(test)

        train = self.process_imp_data(train_imp, all_cols, to_encode, num_cols, data[0])
        test = self.process_imp_data(test_imp, all_cols, to_encode, num_cols, data[1])

        # round categorical numerical columns to nearest category
        train, test = imputer.round_cat_cols(train, test, data, num_cols)
        # round int columns
        train, test = self.round_int(train, test, int64_cols)
        train, test = self.original_types((train, test), data)

        train[target] = train_target
        test[target] = test_target
        return (train, test), None

    def params_callback(self):
        """Sets number of neighbours"""

        @self.app.callback(
            Output("imputation-params", "data", allow_duplicate=True),
            Input('knn-slider', 'value'),
            prevent_initial_call=True
        )
        def set_val(value):
            self.params["neighbors"] = value
            return dash.no_update

