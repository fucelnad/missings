import base64
import dash
import pandas as pd
from dash import html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import urllib

import src.imputation.imputer as imputer
from src.imputation.template_imputer import Imputer
from src.imputation.gain import GainClass


class GAINImputation(Imputer):

    def __init__(self, app):
        super().__init__(app)
        self.name = "GAIN"
        self.params = {"batch_size": 128, "hint_rate": 0.9, "alpha": 100, "iter": 200}

    def provide_info(self):

        encoded_image = base64.b64encode(open("src/program_files/gain.png", "rb").read())
        encoded_image = "data:image/png;base64," + urllib.parse.quote(encoded_image)
        image = html.Img(src=encoded_image, style={"height": "40%", "width": "40%", "display": "block",
                                                   "margin-left": "auto", "margin-right": "auto",
                                                   "margin-top": "10px", "margin-bottom": "15px"})
        text = html.Div([
            html.P("""
            Generative adversarial imputation networks (GAIN) are an example of a deep learning method. GAIN consist of 
            two main components: a generator G and a discriminator D. Both G and D are neural networks. The purpose of 
            G is to impute missing values based on the observed values. The purpose of D is to tell which values have 
            been imputed."""),
            html.P("""
            The generator takes as input a data vector, a random vector and a mask vector. The data vector 
            represents the original data, with missing values indicated by a special symbol. The mask vector indicates 
            which components of the original data are observed and the random vector is a noise variable. It outputs 
            a vector of imputations."""),
            html.P("""
            The discriminator takes the imputed vector and the hint vector as inputs. The hint vector is 
            a random variable obtained by randomly selecting a certain portion of the values in the mask vector 
            to reveal to D. It outputs a vector where the i-th component corresponds to the probability that the 
            i-th component of the imputed vector was observed."""),
            html.P("""
            So D is trained to be as precise as possible, while G is trained to trick D into being imprecise."""),
            html.P(["""
            Features with float, int and category data types that can be cast to numeric data types are imputed as 
            described. However, int data types are rounded to the nearest integer at the end. Numeric features set as 
            category are rounded to the nearest category. Features with category data types that cannot be cast to 
            numeric data types are imputed using mode and one-hot encoded for the purpose of training G and D. [""",
                   html.A("1", href="https://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf", target="_blank"),
                    ", ",
                    html.A("2", target="_blank",
                           href="https://www.sciencedirect.com/science/article/pii/S0957417423007030?via%3Dihub"),", ",
                    html.A("3", target="_blank",
                           href="https://www.ijcai.org/proceedings/2019/0429.pdf"),
                    "]"]),
            image,
            html.P(["MICE imputation [",
                    html.A("1", target="_blank",
                           href="https://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf"),
                    "]."], style={"textAlign": "center"})
        ], style={"margin-left": "100px", "margin-right": "100px", "textAlign": "justify"})
        return text

    def param_tuning(self):
        """Options for number parameters: batch size, hint rate, alpha and number of iterations"""

        batch_sizes = [i for i in [1, 16, 32, 64, 128, 256] if i <= self.rows]
        batch_tmp = max(batch_sizes)

        hint_rates = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        alpha = [40, 60, 80, 100, 120, 140]
        num_iter = [200, 400, 600, 800, 1000, 1200]

        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.H5("Select batch size:", style={"margin-left": "10px"}),
                    dcc.RadioItems(id='gain-batch', options=[{'label': str(i), 'value': i} for i in batch_sizes],
                                   value=batch_tmp, style={"margin-left": "20px"}),
                ], style={"max-width": "100%"}),
                dbc.Col([
                    html.H5("Select hint rate:", style={"margin-left": "10px"}),
                    dcc.RadioItems(id='gain-hint', options=[{'label': str(i), 'value': i} for i in hint_rates],
                                   value=0.9, style={"margin-left": "20px"}),
                ], style={"max-width": "100%"}),
                dbc.Col([
                    html.H5("Select alpha:", style={"margin-left": "10px"}),
                    dcc.RadioItems(id='gain-alpha', options=[{'label': str(i), 'value': i} for i in alpha],
                                   value=100, style={"margin-left": "20px"}),
                ], style={"max-width": "100%"}),
                dbc.Col([
                    html.H5("Select number of iterations:", style={"margin-left": "10px"}),
                    dcc.RadioItems(id='gain-iter', style={"margin-left": "20px"},
                                   options=[{'label': str(i), 'value': i} for i in num_iter], value=1000),
                ], style={"max-width": "100%"}),
            ], style={"max-width": "100%", "overflow-x": "hidden"})
        ], style={"margin-left": "100px", "margin-right": "100px"})

    def get_nearest_cat(self, train_orig, mask_train, mask_test):
        for col in train_orig.columns:
            if train_orig[col].isnull().any() and train_orig[col].dtype == "category" and col in self.num_cols:
                self.train[col] = self.train[col].astype('float64')
                self.test[col] = self.test[col].astype('float64')
                unique_values = train_orig[col].unique()

                self.train.loc[mask_train[col], col] = self.train.loc[mask_train[col], col].apply(
                    lambda x: imputer.find_nearest(unique_values, x))
                self.test.loc[mask_test[col], col] = self.test.loc[mask_test[col], col].apply(
                    lambda x: imputer.find_nearest(unique_values, x))

                self.train[col] = self.train[col].round(decimals=2).astype('category')
                self.test[col] = self.test[col].round(decimals=2).astype('category')

        return self.train, self.test

    def impute_data(self, data, int64_cols, target):
        """Imputes data with GAIN imputation."""

        if not data[0].isna().any().any() and not data[1].isna().any().any():
            return (data[0], data[1]), None

        self.train, self.test = data[0].copy(), data[1].copy()
        # drop target before imputation
        train_target = self.train[target]
        test_target = self.test[target]
        self.train.drop(target, axis=1, inplace=True)
        self.test.drop(target, axis=1, inplace=True)
        cols_orig = self.train.columns

        # encode categorical columns that are not numerical
        self.num_cols, self.non_num_cols = imputer.get_numeric_features(self.train)
        mask_train = self.train[self.num_cols].apply(pd.isnull)
        mask_test = self.test[self.num_cols].apply(pd.isnull)

        encoder = self.handle_categorical()
        cols = self.train.columns

        # GAIN
        gain = GainClass(self.params)
        gain.fit(self.train.values)
        self.train = pd.DataFrame(gain.transform(self.train.values), columns=cols)
        self.test = pd.DataFrame(gain.transform(self.test.values), columns=cols)

        # decode encoded categorical columns
        if encoder:
            self.decode_categorical(encoder, data[1])

        self.train = pd.DataFrame(self.train, columns=cols_orig)
        self.test = pd.DataFrame(self.test, columns=cols_orig)

        # round categorical numerical columns and integer columns
        self.train, self.test = self.original_types((self.train, self.test), data)
        self.get_nearest_cat(data[0], mask_train, mask_test)
        self.train, self.test = self.round_int(self.train, self.test, int64_cols)

        self.train[target] = train_target
        self.test[target] = test_target
        return (self.train, self.test), None

    def params_callback(self):
        """Sets parameters"""

        @self.app.callback(
            Output("imputation-params", "data", allow_duplicate=True),
            Input('gain-batch', 'value'),
            Input('gain-hint', 'value'),
            Input('gain-alpha', 'value'),
            Input('gain-iter', 'value'),
            prevent_initial_call=True
        )
        def set_val(batch, hint, alpha, iterations):
            self.params["batch_size"] = batch
            self.params["hint_rate"] = hint
            self.params["alpha"] = alpha
            self.params["iter"] = iterations

            return dash.no_update

