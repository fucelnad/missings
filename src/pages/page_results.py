import dash
from dash import dash_table, dcc, html
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash.dependencies import Input, Output

from src.pages.page_template import Page


class ResultsPage(Page):

    def __init__(self, app, imputer):
        self.original_data = None
        self.imputer = imputer

        super().__init__(app)

    def create_basic_layout(self):
        """Creates initial layout of the page"""

        layout = html.Div([
            html.Div(id="res-text",  style={"margin": "10px", "display": "flex"}),
            dcc.Loading(
                id="loading-res",
                type="default",
                fullscreen=True,
                children=html.Div(id="res-output",)
            )
        ])

        return layout

    def register_callbacks(self):
        self.show_res_callback()

    def show_res_callback(self):
        @self.app.callback(
            Output("res-text", "children"),
            Output("res-output", "children"),
            Input("imputation-time", "data")
        )
        def show_res(time):
            info = html.Div([
                             html.Div([html.H3("RESULTS OF IMPUTATION", style={"margin-top": "20px"})],
                                      style={"display": "flex", "justify-content": "center"}),
                             html.Br(),
                             html.P("The results page shows a comparison of the distributions of the imputed features based on the training sets only. "
                                    "For numerical features, a histogram is used; for categorical features, a bar plot "
                                    "or data table is displayed, depending on the number of categories. "
                                    "In case of original dataset, observed data points are shown only. In case of imputed dataset both original and imputed points are shown.",
                                    style={"textAlign": "justify"}),
                            html.P("To see how the distribution of data changes using different methods or "
                                   "hyperparameters go back to Imputation page, impute data and return here afterwards. To train "
                                   "machine learning models on currently imputed data move to the page Evaluation.",
                                   style={"textAlign": "left"})],
                            style={"display": "flex", "flex-direction": "column", "align-items": "left",
                                   "justify-content": "center", "margin-left": "100px", "margin-right": "100px"})

            if not self.imputer.original_data or not self.imputer.imputed_data:
                info.children.extend([
                    html.Br(),
                    html.H3("Upload the dataset first and impute missing values", style={"textAlign": "center"})
                ])
                return info, None

            if "res" in self.layouts and not self.imputer.changed_imputation:
                return None, self.layouts["res"]

            self.imputer.changed_imputation = False
            intro = html.Div([html.Br(),
                              html.H5(f"Imputation method used: {self.imputer.method}",
                                      style={"margin-left": "100px"}),
                              html.H5(f"Imputation time: {time}s", style={"margin-left": "100px"})])
            graphs = self.compare_imputation(self.imputer.original_data[0], self.imputer.imputed_data[0])
            self.layouts["res"] = html.Div([info, intro, graphs])

            return None, self.layouts["res"]

    @staticmethod
    def combine_graphs(trace1, trace2, trace3, trace4):
        """Combines more plots into one"""

        fig = make_subplots(rows=1, cols=2)
        fig.add_trace(trace2, row=1, col=1)
        fig.add_trace(trace1, row=1, col=1)
        fig.add_trace(trace3, row=1, col=2)
        fig.add_trace(trace4, row=1, col=2)
        return fig

    @staticmethod
    def plot_numerical(original, imputed, column):
        """Compare numerical column in original and imputed dataset"""

        # trace 2 and trace 3 are used so the y-axis of both graphs will remain the same
        trace1 = go.Histogram(x=original[column], name='Original dataset', nbinsx=50, marker=dict(color='#2e9cf4'))
        trace2 = go.Histogram(x=imputed[column], name='', opacity=0, hoverinfo="none", nbinsx=50)

        trace3 = go.Histogram(x=original[column], name='', opacity=0, hoverinfo="none", nbinsx=50)
        trace4 = go.Histogram(x=imputed[column], name='Imputed dataset', nbinsx=50, marker=dict(color='#bd5104'))

        fig = ResultsPage.combine_graphs(trace1, trace2, trace3, trace4)
        fig.update_layout(title_text=f'Histogram of {column}', barmode='overlay', xaxis_title=column,
            yaxis_title="Frequency", xaxis2_title=column, yaxis2_title="Frequency")

        return html.Div(children=dcc.Graph(figure=fig))

    @staticmethod
    def get_table(original, imputed, column):
        """Compares categorical columns with many categories in original and imputed dataset"""

        original_count_df = original[column].value_counts().reset_index(name='Original Dataset')
        imputed_count_df = imputed[column].value_counts().reset_index(name='Imputed Dataset')

        combined_df = pd.concat([original_count_df.set_index(column), imputed_count_df.set_index(column)], axis=1)
        combined_df = combined_df.fillna(0).reset_index()
        combined_df.columns = [column, 'Original Dataset', 'Imputed Dataset']
        mask = (combined_df['Original Dataset'] != 0) | (combined_df['Imputed Dataset'] != 0)
        combined_df = combined_df[mask]
        combined_df['Increased Frequency'] = (combined_df['Imputed Dataset'] - combined_df['Original Dataset']) / \
                                             combined_df['Original Dataset']
        combined_df['Increased Frequency'] = combined_df['Increased Frequency'].round(2)
        combined_df.sort_values(by='Increased Frequency', ascending=False, inplace=True)

        table = html.Div(children=[
            html.H5(f'Table of {column}'),
            dash_table.DataTable(
                columns=[
                    {'name': column, 'id': column},
                    {'name': 'Original Dataset', 'id': 'Original Dataset'},
                    {'name': 'Imputed Dataset', 'id': 'Imputed Dataset'},
                    {'name': 'Increased Frequency', 'id': 'Increased Frequency'}
                ],
                data=combined_df.to_dict('records'),
                page_size=10,
                style_table={'overflowX': 'auto', 'padding-left': '10px'}
            )
        ], style={'margin-left': '2%', 'margin-right': '2%'})

        return html.Div(children=[table])

    @staticmethod
    def get_bar_plot(original, imputed, column):
        """Compares categorical columns with few categories in original and imputed dataset"""

        unique = pd.concat([original[column], imputed[column]]).unique()

        original_counts = original[column].value_counts().sort_index()
        imputed_counts = imputed[column].value_counts().sort_index()

        trace1 = go.Bar(x=original_counts.index, y=original_counts.values, name='Original dataset',
                        marker=dict(color='#2e9cf4'))
        trace2 = go.Bar(x=imputed_counts.index, y=imputed_counts.values, name='', opacity=0, hoverinfo="none")

        trace3 = go.Bar(x=original_counts.index, y=original_counts.values, name='', opacity=0, hoverinfo="none")
        trace4 = go.Bar(x=imputed_counts.index, y=imputed_counts.values, name='Imputed dataset',
                        marker=dict(color='#bd5104'))

        fig = ResultsPage.combine_graphs(trace1, trace2, trace3, trace4)
        fig.update_layout(
            title_text=f'Bar Chart of {column}',
            barmode='overlay',
            xaxis=dict(tickmode='array', tickvals=unique),
            xaxis2=dict(tickmode='array', tickvals=unique),
            xaxis_title=column,
            yaxis_title="Count",
            xaxis2_title=column,
            yaxis2_title="Count"
        )
        return html.Div(children=dcc.Graph(figure=fig))

    @staticmethod
    def plot_categorical(original, imputed, column):
        """Returns visualization for given column"""

        unique_vals = pd.concat([original[column], imputed[column]]).unique()

        if len(unique_vals) > 25:
            return ResultsPage.get_table(original, imputed, column)
        else:
            return ResultsPage.get_bar_plot(original, imputed, column)

    @staticmethod
    def compare_imputation(original, imputed):
        """Compare original with imputed dataset"""

        graphs = []

        for col in original.columns:
            if not original[col].isna().any():  # compare only imputed values
                continue
            if pd.api.types.is_numeric_dtype(original[col]):
                graphs.append(ResultsPage.plot_numerical(original, imputed, col))
            else:
                graphs.append(ResultsPage.plot_categorical(original, imputed, col))

        if not graphs:
            graphs.append(html.Br())
            graphs.append(html.H5("Nothing imputed due to no missing values present.", style={"margin": "100px"}))

        return html.Div(graphs, style={"margin-left": "30px", "margin-right": "30px"})
