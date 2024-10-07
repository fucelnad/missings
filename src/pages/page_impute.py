from dash import dcc, html
import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import time

# add file where the class implementing new method is placed
from src.imputation.gain_imputer import GAINImputation
from src.pages.page_template import Page
from src.imputation.knn_imputer import KNNImputation
from src.imputation.mean_imputer import MeanImputation
from src.imputation.mice_imputer import MICEImputation
from src.imputation.missforest_imputer import MissForestImputation


class ImputePage(Page):

    def __init__(self, app, train):
        self.original_data = None
        self.imputed_data = None
        self.target = None
        self.train_page = train

        self.imputer = None
        self.method = None
        self.changed_imputation = False

        # new imputation method can be added here
        # label is displayed in the dropdown menu
        # value is used internally as an id of the method
        self.options = [
            {"label": "Mean Imputation", "value": "Mean"},
            {"label": "MissForest", "value": "MissForest"},
            {"label": "Imputation using kNN", "value": "kNN"},
            {"label": "MICE", "value": "MICE"},
            {"label": "GAIN", "value": "GAIN"}
        ]

        # add implementation of method
        # key in the dictionary has to be the same as previously chosen method id
        self.methods = {
            "MissForest": MissForestImputation(app),
            "Mean": MeanImputation(app),
            "kNN": KNNImputation(app),
            "MICE": MICEImputation(app),
            "GAIN": GAINImputation(app)
        }

        super().__init__(app)

    def update_data(self, data):
        """Updates data when uploaded"""

        self.original_data = data
        self.method = None
        self.imputer = None
        self.imputed_data = None

    def update_target(self, target):
        """Updates data when uploaded"""

        self.target = target
        self.method = None
        self.imputer = None
        self.imputed_data = None

    def create_basic_layout(self):
        """Creates initial layout of the page"""

        layout = html.Div([
            html.Div(id="impute-dropdown"),
            html.Div(id="impute-output"),
            dcc.Loading(
                id="loading-status",
                type="default",
                fullscreen=True,
                children=html.Div(id="impute-status")
            ),
            html.Div(id="impute-info")
        ])

        return layout

    def params_callback(self):
        @self.app.callback(
            Output("impute-output", "children"),
            Output("impute-info", "children"),
            [Input("impute-dropdown", "value")]
        )
        def get_params(method):
            if method not in self.methods:
                return dash.no_update, dash.no_update

            self.method = method
            self.imputer = self.methods[method]
            text = self.imputer.provide_info()
            if not self.original_data or not self.target:
                return html.Div([html.H3("Upload the dataset first, choose its target and type of machine learning problem",
                            style={"textAlign": "center"})]), ""
            self.imputer.rows = len(self.original_data[0])
            buttons = self.imputer.param_tuning()

            return html.Div([
                buttons,
                html.Button('Confirm', id='conf', n_clicks=0,
                            style={"margin-left": "100px", "margin-top": "10px", "margin-bottom": "10px"}),
                html.Div(id="impute-graphs")
            ]), text

    def impute_callback(self):
        @self.app.callback(
            Output("imputation-time", "data"),
            Output("impute-status", "children"),
            Input("conf", "n_clicks"),
            State("int64-cols", "data"),
        )
        def impute(n_clicks, int64_cols):
            if n_clicks > 0:
                self.changed_imputation = True
                start = time.time()
                try:
                    self.imputed_data, modal = self.imputer.impute_data(self.original_data, int64_cols, self.target)
                except Exception as e:
                    modal = dbc.Modal([
                        dbc.ModalHeader(dbc.ModalTitle("Warning: imputation not successful")),
                        dbc.ModalBody(f"Unable to impute data due to error: {e}"),
                    ], id="imputation-modal", is_open=True, size="lg")
                    return 0, modal
                end = time.time()
                imp_time = round(end-start, 2)
                if not modal:
                    modal = dbc.Modal([dbc.ModalHeader(dbc.ModalTitle("Imputation complete"))],
                                      id="imputation-modal", is_open=True, size="lg")
                if self.imputed_data:
                    self.train_page.update_imputed_data(self.imputed_data, self.imputer.name,
                                                        self.imputer.params, imp_time)
                return imp_time, modal

            return [dash.no_update] * 2

    def upload_callback(self):
        """Updates output were new data were uploaded"""

        @self.app.callback(
            [Output("impute-dropdown", "children")],
            [Input("upload-flag", "data")]
        )
        def intro(_):
            info = html.Div([
                html.Div([html.H3("IMPUTATION", style={"margin-top": "20px"})],
                         style={"display": "flex", "justify-content": "center"}),
                html.Br(),
                html.P("This page allows you to impute missing values. Select one of the options "
                       "offered and the details of the method will be displayed. Hyperparameter tuning is "
                       "available for most of them. "
                       "The imputation of training and test sets is always based on the information available"
                       " from the training set only. If the selected method takes too long to impute, you can refresh the "
                       "page and select different method.", style={"textAlign": "justify"}),
                html.P("To see what values were imputed move to the page named Results of Imputation, where the "
                       "changes are presented through visualizations."
                       " To train machine learning models move to the page named Evaluation.", style={"textAlign": "left"}),
                html.Br(),
            ], style={"display": "flex", "flex-direction": "column", "align-items": "left",
                      "justify-content": "center", "margin-right": "100px"})

            if self.original_data is None or self.target is None:
                info.children.extend([
                    html.H3("Upload the dataset first, choose its target and type of machine learning problem",
                            style={"textAlign": "center"})
                ])
                return [html.Div([info], style={"margin-left": "100px"})]

            return [html.Div([
                info,
                html.H3("Pick an imputation method"),
                dcc.Dropdown(
                    id="impute-dropdown",
                    options=self.options,
                    placeholder="Select an option",
                    style={"width": "500px"},
                ),
            ], style={"margin-left": "100px", "margin-bottom": "20px"})]

    def register_callbacks(self):
        """Handles application callbacks"""

        self.upload_callback()
        self.params_callback()
        for method in self.methods.values():
            method.params_callback()
        self.impute_callback()
