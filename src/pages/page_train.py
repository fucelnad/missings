import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

# add file where the class implementing new model is placed
from src.ml_training.knn_regr import kNNRegressor
from src.ml_training.knn_class import kNNClass
from src.pages.page_template import Page
from src.ml_training.svm_class import SVMClass
from src.ml_training.svr_regr import SVMRegressor
from src.ml_training.tree_class import TreeClass
from src.ml_training.tree_regr import TreeRegressor


class TrainPage(Page):
    def __init__(self, app):
        super().__init__(app)
        self.original_data, self.imputed_data = None, None
        self.imp_method, self.imp_params = None, None
        self.target, self.ml_type = None, None

        self.train_orig = {}
        self.imp_time = {}
        self.used_algo = {}

        # new ML model can be added here
        # label is displayed in the dropdown menu
        # value is used internally as an id of the model
        self.classification = [{"label": "Decision tree", "value": "tree_class"},
                               {"label": "Support vector machine", "value": "svm_class"},
                               {"label": "K nearest neighbours", "value": "knn_class"}]
        self.regression = [{"label": "Decision tree", "value": "tree_regr"},
                           {"label": "Support vector machine", "value": "svm_regr"},
                           {"label": "K nearest neighbours", "value": "knn_regr"}]

        # add implementation of model
        # key in the dictionary has to be the same as previously chosen model id
        self.algo_impl = {
            "tree_class": TreeClass(),
            "tree_regr": TreeRegressor(),
            "svm_class": SVMClass(),
            "svm_regr": SVMRegressor(),
            "knn_regr": kNNRegressor(),
            "knn_class": kNNClass()
        }

    def update_target(self, target, ml_type):
        self.target = target
        self.ml_type = ml_type
        self.clear_results()

    def update_original_data(self, data):
        """Updates original data when uploaded"""

        self.layouts["train"] = None
        self.original_data, self.imputed_data = data, None
        self.train_orig = {}

    def clear_results(self):

        if self.ml_type == "regression":
            self.used_algo = {"Model": [], "Imputation": [], "Hyperparams": [],
                              "Test RMSE": [], "Train RMSE": []}
        else:
            self.used_algo = {"Model": [], "Imputation": [], "Hyperparams": [],
                              "Test Accuracy": [], "Train Accuracy": []}

        self.imp_time = {"Imputation": [], "Hyperparams": [], "Time": []}

    def update_imputed_data(self, data, method, params, time):
        """Updates imputed data"""

        self.imputed_data = data
        self.imp_method = method
        self.imp_params = params
        self.imp_time["Imputation"].append(method)
        self.imp_time["Hyperparams"].append(params.copy())
        self.imp_time["Time"].append(time)
        self.layouts["train"] = None

    def create_basic_layout(self):
        """Creates initial layout of the page"""

        layout = html.Div(children=[
            html.Div(id="ml-dropdown", style={"margin": "10px", "display": "flex"}),
            dcc.Loading(
                id="loading-train",
                type="default",
                fullscreen=True,
                children=html.Div(id="train-status")
            ),
            html.Div(id="train-output", style={"margin": "10px"})
        ])
        return layout

    def register_callbacks(self):
        """Handles application callbacks"""

        self.upload_callback()
        self.model()

    def upload_callback(self):
        """Callback for newly uploaded data"""

        @self.app.callback(
            Output("ml-dropdown", "children"),
            Input("upload-flag", "data")
        )
        def pick_algo(_):
            """Dropdowns with ML algorithms"""

            info = html.Div([
                html.Div([html.H3("EVALUATION", style={"margin-top": "20px"})],
                         style={"display": "flex", "justify-content": "center"}),
                html.Br(),
                html.P("Machine learning models can be trained here. After selecting one of the models, it is "
                       "trained on the original and imputed training sets. The results obtained on the original and imputed"
                       " test sets are then displayed. Note that the original dataset will only remain unchanged if the"
                       " selected model can handle missing values. Otherwise, missing values are imputed with -1 and"
                       " a feature is added to indicate whether it was filled or not. You can choose more machine "
                       "learning models and observe the changes in all of them.",
                       style={"textAlign": "justify"}),
                html.P("To see how machine learning models change using different methods or hyperparameters used "
                       "during imputation go back to the Imputation page and select one. Once the imputation is completed"
                       " return here and train models. Remember that results of all the imputation methods and machine "
                       "learning algorithms used can be seen on the Summary page.", style={"textAlign": "justify"})
            ], style={"display": "flex", "flex-direction": "column", "align-items": "left",
                      "justify-content": "center", "margin-right": "100px", "margin-left": "100px"},
            )

            if not self.original_data or not self.target or not self.ml_type:
                info.children.extend([
                    html.Div([
                     html.H3("Upload the dataset first, choose its target and type of machine learning problem",
                             style={"display": "flex", "justify-content": "center"})
                ])])
                self.layouts["train"] = html.Div([])
                return info

            ml = self.create_ml_dropdown(self.ml_type)
            return html.Div([info, ml])

    def create_ml_dropdown(self, ml_type):
        """Helper function to create ML dropdown"""

        if ml_type == "classification":
            options = self.classification
        else:
            options = self.regression

        return html.Div([
            html.H4("Pick a machine learning algorithm to train", style={"margin": "10px"}),
            dcc.Dropdown(
                id="ml-dropdown",
                options=options,
                placeholder="Select an algorithm",
                style={"width": "600px"}
            )
        ], style={"margin-left": "100px"})

    def model(self):
        """Models with original and imputed data trained"""

        @self.app.callback(
            Output("train-output", "children"),
            Output("train-status", "children"),
            Input("ml-dropdown", "value")
        )
        def train_model(algorithm):
            if algorithm is None:
                if "train" in self.layouts:
                    return self.layouts["train"], dash.no_update
                return [], dash.no_update

            if not self.original_data or not self.target or not self.ml_type:
                return html.H3("Upload the dataset first, choose its target and type of machine learning problem",
                               style={"textAlign": "center"})
            method = self.algo_impl[algorithm]
            try:
                output = method.handle_models(self, self.target)
            except Exception as e:
                modal = dbc.Modal([
                    dbc.ModalHeader(dbc.ModalTitle("Warning")),
                    dbc.ModalBody(f"Unable to train model due to error: {e}"),
                ], id="train-modal", is_open=True, size="lg")
                return [], modal
            self.layouts["train"] = html.Div([html.H5(f"Previously chosen algorithm: {method.model}"),
                                              output], style={"border": "1px solid #ddd", "border-radius": "5px",
                                                              "padding": "10px", "background-color": "#f9f9f9",
                                                              "margin": "10px"})

            return output, dash.no_update
