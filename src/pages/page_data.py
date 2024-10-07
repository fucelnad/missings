import ast
import base64
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, ALL
import dash_bootstrap_components as dbc
import numpy as np
import io
import pandas as pd

from src.data_loading import data_handler as dh
from src.pages.page_template import Page

MAX_SIZE = 100  # sets max file size to 100 MB
MAX_CLASS = 10  # sets max allowed number of categories for classification

class DataPage(Page):
    """Class representing a page which enables user to upload data"""

    def __init__(self, app, stats, impute, train):
        super().__init__(app)

        self.data = None
        self.filename = None
        self.target = None
        self.problem = None
        # used for storing target and problem when the user is about to decide whether to delete missing rows
        self.storage = (None, None)

        self.stats_page = stats
        self.impute_page = impute
        self.train_page = train

    def create_basic_layout(self):
        """Creates initial layout of the page"""

        upload = dh.upload_button(MAX_SIZE)
        layout = html.Div([
            upload,
            html.Div(style={"display": "flex", "flex-direction": "row"}, children=[
                html.Div(id="target", style={"flex": "50%", "padding": "10px"}),
                html.Div(id="data-setting", style={"flex": "50%", "padding": "10px"}),
            ]),
            html.Div(id="warn-del"),  # used for warning message

            # used for the loading animation when new dataset is being uploaded
            dcc.Loading(
                id="loading-output",
                type="default",
                fullscreen=True,
                children=html.Div(id="data-wait")
            ),
            html.Div(id="data-output")
        ])
        return layout

    def parse_contents(self, contents, filename):
        """Parse the uploaded content and set the dataframe"""

        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        self.filename = filename

        try:
            if "csv" in filename:
                self.data = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
                if len(self.data.columns) < 3 or self.data.shape[0] < 30:
                    self.filename = None
                    self.data = None
                    return html.Div([f"There was an error processing {filename}. "
                                     f"Dataset with at least 3 columns and 30 rows must be given."])
                for col in self.data.columns:
                    if self.data[col].dtype in ['object', 'bool']:  # put all the object and bool data types to category
                        self.data[col] = self.data[col].astype('category')
        except Exception as e:
            self.filename = None
            print(f"Error processing {filename}: {e}")
            return html.Div([f"There was an error processing {filename}."])

        return html.Div([f"Incorrect file uploaded. Allowed format is CSV and size no more than 100 MB."])

    def reset_data(self):
        """Resets the data attributes"""

        self.data = None
        self.problem = None
        self.target = None

    def process_uploaded_data(self, list_of_contents, list_of_names):
        """Saves uploaded data"""

        total_size = sum(len(content) for content in list_of_contents)
        self.reset_data()  # delete old dataset
        err = html.Div([f"Incorrect file uploaded. Allowed format is CSV and size no more than 100 MB."])
        try:
            if total_size > MAX_SIZE * 1024 * 1024:
                err = html.Div([f"The uploaded file is too big. Allowed size is 100 MB."])
                raise ValueError

            for c, n in zip(list_of_contents, list_of_names):
                err = self.parse_contents(c, n)
                if self.data is None:  # incorrect file given
                    raise ValueError
                else:
                    self.layouts["data"] = dh.data_preview(self.data, self.filename, self.target, self.problem)

        except ValueError:
            self.layouts["data"] = dh.display_error_message(err)

        self.update_pages(self.data)

        return self.layouts["data"]

    def update_pages(self, data, update_all=False):
        """Update data to other pages"""

        split_data = dh.split_data(data)
        self.stats_page.update_data(split_data)
        self.train_page.clear_results()
        if update_all:  # split data to train and test set and upload to impute and train pages
            self.impute_page.update_data(split_data)
            self.train_page.update_original_data(split_data)
        else:
            self.impute_page.update_data(None)
            self.train_page.update_original_data(None)


    def handle_target(self):
        """Displays warn message about dropping rows if target is missing.
        Also warns about the need to drop columns that are missing in more than 80% of observations."""

        num_missing = self.data[self.target].isnull().sum()
        data = self.data.dropna(subset=[self.target])

        missing_percent = data.isnull().sum() / len(data)
        self.missing_cols = missing_percent[missing_percent >= 0.8].index.tolist()
        text_cols, text = "", ""

        if not self.missing_cols and num_missing == 0:
            return None

        self.storage = (self.target, self.problem)
        self.target, self.problem = None, None
        self.train_page.update_target(None, None)
        self.impute_page.update_target(None)

        if data.shape[0] < 30:
            text = (f"Target is missing in too many rows which would result in dataset having less than 30 rows. "
                    f"Choose another target.")
            return dh.create_warn_modal(html.Div([html.P(text), html.P(text_cols)]), "target-rows")

        if self.missing_cols:

            if len(self.data.columns) - len(self.missing_cols) < 3:
                text = (f"Too many columns have more than 80 % values missing and would have to be dropped. "
                        f"However, to move to the next steps at least 3 column must be present. "
                        f"Pick another target or upload different dataset.")
                return dh.create_warn_modal(html.Div([html.P(text), html.P(text_cols)]), "target-rows")

            text_cols = (f"Columns with more than 80 % missing values (after the removal of NaNs from target) "
                         f"will have to be dropped. Those are: {self.missing_cols}.")

        if num_missing != 0:
            text = (f"Target is missing in {num_missing} rows which have to be deleted so it will be possible to train "
                    f"machine learning models. Press CONFIRM to do so, otherwise pick another target.")

        modal = dh.create_warn_modal(html.Div([html.P(text), html.P(text_cols)]), "del-cols")
        modal.children.append(dbc.ModalFooter(dbc.Button("Confirm", id="target-del", n_clicks=0)))

        return modal

    def update_options(self, n_clicks, target, problem, int64_cols):
        """Updates selected target and ML problem. Passes data to impute and train page."""

        if n_clicks is None or n_clicks == 0:
            return [dash.no_update] * 4

        self.target, self.problem = target, problem
        modal = None

        try:
            if not target or not problem:
                modal = dh.create_warn_modal("Both target and type of problem have to be selected.", "select-both")
                return dash.no_update, modal, dash.no_update, dash.no_update
            modal = self.handle_target()

            if self.data[target].notnull().sum() < 2:
                modal = dh.create_warn_modal(f"At least two values of target must be observed. Number of "
                                             f"observed values in chosen target is {self.data[target].notnull().sum()}."
                                             , "target-option")
                raise ValueError

            if problem == "regression" and not pd.api.types.is_numeric_dtype(self.data[target]):
                modal = dh.create_warn_modal("In regression target has to be numerical.", "regression")
                raise ValueError

            if problem == "classification":
                class_num = len(self.data[target].unique())
                if self.data[target].isnull().any():
                    class_num -= 1
                if class_num > MAX_CLASS:
                    modal = dh.create_warn_modal(f"Maximum allowed number of unique classes in classification is "
                                                 f"set to {MAX_CLASS}. Number of classes in chosen target is {class_num}.",
                                                 "classification")
                    raise ValueError

        except ValueError:
            self.target, self.problem = None, None
            self.train_page.update_target(None, None)
            self.impute_page.update_target(None)
            self.layouts["data"] = dh.data_preview(self.data, self.filename)
            return "change", modal, self.layouts["data"], dh.select_data_types(self.data, int64_cols)

        self.update_pages(self.data, True)
        self.train_page.update_target(self.target, self.problem)
        self.impute_page.update_target(self.target)
        self.layouts["data"] = dh.data_preview(self.data, self.filename, self.target, self.problem)

        return "change", modal, self.layouts["data"], dh.select_data_types(self.data, int64_cols)

    def upload_data_callback(self):
        """Callback for uploaded data"""

        @self.app.callback(
            Output("target", "children"),
            Output("data-setting", "children"),
            Output("data-output", "children", allow_duplicate=True),
            Output("data-wait", "data"),  # used for loading animation
            Output("upload-data", "contents"),  # so that it is possible to load the same data more times
            Output("int64-cols", "data"),
            Input("upload-data", "contents"),
            State("upload-data", "filename"),
            State("int64-cols", "data"),
        )
        def upload_data(list_of_contents, list_of_names, int64_col):
            """Detects upload of new dataset"""

            if list_of_contents is None and "data" not in self.layouts:
                return [dash.no_update] * 6

            data_table = None
            if list_of_contents is not None:
                data_table = self.process_uploaded_data(list_of_contents, list_of_names)
                int64_col = []
            elif "data" in self.layouts:
                data_table = self.layouts["data"]

            target = dh.select_target(self.data)
            settings = dh.select_data_types(self.data, int64_col)
            return target, settings, data_table, None, None, int64_col

        @self.app.callback(
            Output("datatable-paging", "page_count"),
            Output("datatable-paging", "data"),
            Input("datatable-paging", "page_current"),
            Input("datatable-paging", "page_size"),
            Input("datatable-paging", "filter_query"))
        def update_table(page_current, page_size, filter_used):
            """Detects scrolling in data preview"""

            return dh.filter_data(self.data, page_current, page_size, filter_used)

        @self.app.callback(
            Output('upload-flag', 'data'),
            Output('warn-del', 'children'),
            Output('data-output', 'children'),
            Output('data-setting', "children", allow_duplicate=True),
            Input('confirm-button', 'n_clicks'),
            State('target-dropdown', 'value'),
            State('problem-button', 'value'),
            State("int64-cols", "data"),
            prevent_initial_call=True
        )
        def update_options_callback(n_clicks, target, problem, int64_cols):
            return self.update_options(n_clicks, target, problem, int64_cols)

        @self.app.callback(
            Output('upload-flag', 'data', allow_duplicate=True),
            Output("warn-del", "children", allow_duplicate=True),
            Output("data-setting", "children", allow_duplicate=True),
            Output("int64-cols", "data", allow_duplicate=True),
            Input({'type': 'setting', 'index': ALL}, 'value'),
            State("int64-cols", "data"),
            prevent_initial_call=True
        )
        def update_types(data_types, int64_cols):

            ctx = dash.callback_context
            ret, change = dh.handle_types_choice(data_types, self.data, ctx, int64_cols)
            if change:  # some data types were changed
                update_all = self.target is not None
                self.update_pages(self.data, update_all)
            return ret

        @self.app.callback(
            Output("warn-del", "children", allow_duplicate=True),
            Output("data-output", "children", allow_duplicate=True),
            Output("target", "children", allow_duplicate=True),
            Output("data-setting", "children", allow_duplicate=True),
            Input("confirm-del", "n_clicks"),
            State("data-columns", "value"),
            State("int64-cols", "data"),
            prevent_initial_call=True
        )
        def delete_cols(n_clicks, col_to_del, int64_cols):
            if n_clicks is None or n_clicks == 0 or col_to_del is None:
                return [dash.no_update] * 4

            if len(col_to_del) >= len(self.data.columns) - 2:
                return dh.create_warn_modal("Can not delete that many columns. At least 3 must stay to train "
                                            "ML models.", "del-col"), dash.no_update, dash.no_update, dash.no_update
            if self.target in col_to_del:
                return (dh.create_warn_modal("Can not delete target column.", "del-col"), dash.no_update,
                        dash.no_update, dash.no_update)

            self.data = self.data.drop(col_to_del, axis=1)
            self.layouts["data"] = dh.data_preview(self.data, self.filename, self.target, self.problem)

            update_all = self.target is not None
            self.update_pages(self.data, update_all)

            return (dash.no_update, self.layouts["data"], dh.select_target(self.data),
                    dh.select_data_types(self.data, int64_cols))

        @self.app.callback(
            Output("target", "children", allow_duplicate=True),
            Output("data-setting", "children", allow_duplicate=True),
            Output("upload-flag", "data", allow_duplicate=True),
            Output("del-cols", "is_open"),
            Output("data-output", "children", allow_duplicate=True),
            Input("target-del", "n_clicks"),
            State("int64-cols", "data"),
            prevent_initial_call=True
        )
        def del_col(n_clicks, int64_cols):
            if n_clicks is None or n_clicks == 0:
                return [dash.no_update] * 5

            self.target = self.storage[0]
            self.problem = self.storage[1]
            self.train_page.update_target(self.target, self.problem)
            self.impute_page.update_target(self.target)
            self.data.drop(columns=self.missing_cols, inplace=True)
            self.data = self.data.dropna(subset=[self.target])

            self.update_pages(self.data, True)
            self.layouts["data"] = dh.data_preview(self.data, self.filename, self.target, self.problem)
            settings = dh.select_data_types(self.data, int64_cols)
            target = dh.select_target(self.data)

            return target, settings, dash.no_update, False, self.layouts["data"]

    def register_callbacks(self):
        """Handles application callbacks"""

        self.upload_data_callback()
