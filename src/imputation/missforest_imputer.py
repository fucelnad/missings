import base64
from dash import dcc, html
import dash
from dash.dependencies import Input, Output
import urllib

import src.imputation.imputer as imputer
from src.imputation.missforest import MissForest
from src.imputation.template_imputer import Imputer


class MissForestImputation(Imputer):

    def __init__(self, app):

        super().__init__(app)
        self.params = {"max_iter": 5}
        self.name = "MissForest"

    def provide_info(self):

        encoded_image = base64.b64encode(open("src/program_files/missforest.png", "rb").read())
        encoded_image = "data:image/png;base64," + urllib.parse.quote(encoded_image)
        image = html.Img(src=encoded_image, style={"height": "70%", "width": "70%", "display": "block",
                                                   "margin-left": "auto", "margin-right": "auto"})
        text = html.Div([
            html.P("MissForest imputation is an iterative method based on random forests."),
            html.P("It consists of the following steps:"),
            html.Ol([
                html.Li(
                    "Initial mean imputation of numerical features and mode imputation of categorical features is "
                    "performed."),
                html.Li([
                    "One of the features with missing values is selected. It consists of missing values y",
                    html.Sub("mis"), " and observed values y", html.Sub("obs"), "."]),
                html.Li([
                    "In case of numerical features, a random forest regressor is trained to predict y", html.Sub("obs"),
                    " based on x", html.Sub("obs"), ". In case of categorical features, a random forest classifier "
                                                    "is trained."]),
                html.Li([
                    "The trained random forest is used to predict y", html.Sub("mis"), " based on x", html.Sub("mis"),
                    ". These predictions are then imputed."])]),
            html.P("""
                Steps 2-4 are repeated for each feature with missing values, forming an iteration. Iterations are 
                repeated until the difference between the newly imputed data matrix and the previous one increases 
                with respect to both feature types, if present"""),
            html.P(["""
            Features are imputed as described. However, in case of feature with int data type it is rounded to the 
            closest integer at the end. [""",
                   html.A("1", target="_blank",
                          href="https://academic.oup.com/bioinformatics/article/28/1/112/219101?login=false"),
                   ", ",
                    html.A("2", target="_blank",
                           href="https://www.sciencedirect.com/science/article/pii/S0957417423007030?via%3Dihub"),
                    "]"]),
            image,
            html.P(["MissForest [",
                    html.A("3", target="_blank",
                           href="https://towardsdatascience.com/missforest-the-best-missing-data-imputation-algorithm-4d01182aed3"),
                    "]."], style={"textAlign": "center"})
        ], style={"margin-left": "100px", "margin-right": "100px", "textAlign": "justify"})
        return text

    def param_tuning(self):
        """Options for max number of iterations"""

        return html.Div([
            html.H5("Select maximal number of iterations:"),
            dcc.Slider(
                id='missforest-slider',
                min=5, max=15,
                step=1, value=5
            ),
        ], style={"margin-left": "100px", "margin-right": "100px"})

    def impute_data(self, data, int64_cols, target):

        if not data[0].isna().any().any() and not data[1].isna().any().any():
            return (data[0], data[1]), None

        train, test = data[0].copy(), data[1].copy()
        # drop target before imputation
        train_target = train[target]
        test_target = test[target]
        train.drop(target, axis=1, inplace=True)
        test.drop(target, axis=1, inplace=True)

        cat = imputer.get_categorical_features(train)
        mf = MissForest(self.params["max_iter"])
        mf.fit(
            x=train,
            categorical=cat
        )
        train_imputed = mf.transform(x=train)
        test_imputed = mf.transform(x=test)

        train_imputed, test_imputed = self.original_types((train_imputed, test_imputed), data)
        train_imputed, test_imputed = self.round_int(train_imputed, test_imputed, int64_cols)

        train_imputed[target] = train_target
        test_imputed[target] = test_target
        return (train_imputed, test_imputed), None

    def params_callback(self):
        """Sets max number of iterations"""

        @self.app.callback(
            Output("imputation-params", "data", allow_duplicate=True),
            [Input('missforest-slider', 'value')],
            prevent_initial_call=True
        )
        def set_val(value):
            self.params["max_iter"] = value
            return dash.no_update
