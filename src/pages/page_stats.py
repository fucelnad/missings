from dash import html
from dash.dependencies import Input, Output

from src.pages.page_template import Page
from src.stats import stats_generator


class StatsPage(Page):
    """Class representing a page which enables users to show graphs and basic stats"""

    def __init__(self, app):
        self.data = None  # training set
        self.test = None  # test set
        super().__init__(app)

    def update_data(self, data):
        """Updates data when uploaded"""
        if data is None:
            self.data = None
            self.test = None
        else:
            self.data = data[0]
            self.test = data[1]
        self.layouts["stats"] = None

    def create_basic_layout(self):
        """Creates initial layout of the page"""

        layout = html.Div(children=[
            html.Div(id="stats-info"),
            html.Div(id="column-dropdown", style={"width": "100%"}),
            html.Div(id="column-stats", style={"width": "100%"}),
            html.Div(style={"display": "flex", "flex-direction": "row"}, children=[
                html.Div(id="missing-table", style={"flex": "50%", "padding": "10px"}),
                html.Div(id="corr-matrix", style={"flex": "50%", "padding": "10px", "margin-left": "100px"}),
            ]),
            html.Div(id="missing-graph", style={"width": "100%"}),
        ])

        return layout

    def register_callbacks(self):
        """Handles application callbacks"""

        self.info_callback()
        self.dropdown_callback()

    def info_callback(self):
        """Callback for newly uploaded data"""

        @self.app.callback(
            [Output("stats-info", "children"),
             Output("missing-table", "children"),
             Output("corr-matrix", "children"),
             Output("missing-graph", "children"),
             Output("column-dropdown", "children")],
            Input('data-target', 'data')
        )
        def update_stats_layout(target):
            """Computes stats"""

            info = html.Div([html.H3("STATISTICS", style={"margin-top": "20px"}),
                             html.Br(),
                             html.P("Once the dataset is uploaded, you can view its statistics "
                                    "here. The percentage of missing values in the training and test sets are displayed. "
                                    "The rest of the page shows only the training set. You can have a look at a matrix plot and a "
                                    "dendrogram. Correlation matrix and plots of features are also shown. The statistics "
                                    "are updated based on the changes in the data.",
                                    style={"textAlign": "justify"})],
                            style={"display": "flex", "flex-direction": "column", "align-items": "center",
                                   "justify-content": "center", "margin-left": "100px", "margin-right": "100px"})

            if self.data is None:
                info.children.extend([
                    html.Br(),
                    html.H3("Upload the dataset first", style={"textAlign": "center"})
                    ])

                self.layouts["stats"] = (info, [], [], [], [])
                return self.layouts["stats"]

            if self.layouts["stats"] is not None:
                return self.layouts["stats"]

            matrix = stats_generator.corr_matrix(self.data)
            table = stats_generator.missing_table(self.data, self.test)
            button = stats_generator.column_button(self.data, target)
            missing = stats_generator.missing_graph(self.data)

            self.layouts["stats"] = info, table, matrix, missing, button
            return self.layouts["stats"]

    def dropdown_callback(self):
        """Detects selected column for visualization"""

        @self.app.callback(
            Output("column-stats", "children"),
            [Input("column-dropdown", "value")]
        )
        def update_selected_column_message(selected_column):
            if selected_column is None:
                return ""

            else:
                return html.Div(style={"display": "flex", "flex-direction": "row", "margin": "10px"}, children=[
                    html.Div(style={"flex": "33%", "padding": "10px"}, children=[
                        stats_generator.col_stats(self.data, selected_column)]),
                    html.Div(style={"flex": "33%", "padding": "10px"}, children=[
                        stats_generator.col_graph(self.data, selected_column)])
                    ])
