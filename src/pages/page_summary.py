import pandas as pd
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objects as go

from src.pages.page_template import Page


class SummaryPage(Page):

    def __init__(self, app, train):
        super().__init__(app)

        self.train_page = train

    def create_basic_layout(self):
        """Creates initial layout of the page"""

        layout = html.Div(children=[
            html.Div(id="summary-text"),
            html.Div(id="summary-output", style={"margin": "10px"}),
            html.Div(id="summary-graph", style={"margin": "10px"}),
            html.Div(id="summary-time", style={"margin": "10px"})
        ])
        return layout

    def register_callbacks(self):
        """Handles application callbacks"""

        self.text_callback()
        self.output_callback()
        self.time_callback()

    @staticmethod
    def add_trace(fig, df, accuracy_type, color, max_len, x_labels):
        labels = df[accuracy_type].apply(lambda x: str(x).rjust(max_len))  # use same number of symbols

        fig.add_trace(go.Bar(
            y=x_labels + df['Hyperparams'],
            x=df[accuracy_type],
            name=accuracy_type,
            customdata=df['Hyperparams'],
            hovertemplate=f'<i>Hyperparams</i><br>%{{customdata}}<br>' +
                          f'<br>' +
                          f'<i>{accuracy_type}</i>: %{{x}}<br>',
            marker_color=color,
            text=labels,
            textposition='outside',
            textfont=dict(
                size=12,
            ),
            orientation='h'  # make bars horizontal
        ))

        return fig

    def create_fig(self, df, metric, x_labels):
        """Returns summary fig grouped by results."""

        fig = go.Figure()
        max_len = len(str(max(df[f'Train {metric}'].max(), df[f'Test {metric}'].max())))
        fig = self.add_trace(fig, df, f'Train {metric}', '#2e9cf4', max_len, x_labels)
        fig = self.add_trace(fig, df, f'Test {metric}', '#bd5104', max_len, x_labels)

        x_axis = dict(range=[0, max(df[f'Train {metric}'].max(), df[f'Test {metric}'].max()) * 1.2])
        if metric == "Accuracy":
            x_axis = dict(
                range=[0, max(df[f'Train {metric}'].max(), df[f'Test {metric}'].max()) * 1.2],
                tickmode='array',
                tickvals=[0, 20, 40, 60, 80, 100],
                ticktext=['0', '20', '40', '60', '80', '100']
            )

        graph_height = max(450, 50 * len(df))
        fig.update_layout(
            barmode='group',
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(x_labels))),
                ticktext=x_labels
            ),
            xaxis=x_axis,
            height=graph_height
        )
        return fig

    def preprocess_data(self, group_by, metric, order):
        df = pd.DataFrame(self.train_page.used_algo)
        asc = True  # sort ascending by metric
        if metric == "RMSE":
            asc = False

        if order == 'name':
            if group_by == 'Model':
                df = df.sort_values(['Model', 'Imputation'])
            else:
                df = df.sort_values(['Imputation', 'Model'])
        elif order == 'train_acc':
            df = df.sort_values([group_by, f'Train {metric}'], ascending=[True, asc])
        elif order == 'test_acc':
            df = df.sort_values([group_by, f'Test {metric}'], ascending=[True, asc])
        df['Hyperparams'] = df['Hyperparams'].apply(str)
        df = df.drop_duplicates(subset=['Model', 'Imputation', 'Hyperparams'], keep='last')

        return df

    def text_callback(self):
        @self.app.callback(
            Output("summary-text", "children"),
            Input("upload-flag", "data")
        )
        def text(_):
            info = html.Div([
                html.H3("SUMMARY", style={"margin-top": "20px"}),
                html.Br(),
                html.P("This page is dedicated to the comparison of the trained machine learning models and the imputation "
                       "techniques used. Classification algorithms are compared by accuracy, while regression algorithms"
                       " by RMSE.", style={"textAlign": "justify"}),
                html.Br(),
            ], style={"display": "flex", "flex-direction": "column", "align-items": "center",
                      "justify-content": "center", "margin-right": "100px", "margin-left": "100px"})

            if (not self.train_page.original_data or
                    (self.train_page.used_algo["Model"] == [] and self.train_page.imp_time["Time"] == [])):
                info.children.extend([html.H3("Upload the dataset first, impute data and train machine learning models")])
                return info

            return info

    def output_callback(self):

        @self.app.callback(
            Output("summary-output", "children"),
            Input("upload-flag", "data")
        )
        def output(_):

            if self.train_page.original_data is None or not self.train_page.used_algo["Model"]:
                return []

            metric = 'RMSE' if self.train_page.ml_type == 'regression' else 'Accuracy'

            ret = html.Div([
                html.H5("Pick order of methods"),
                dcc.Dropdown(
                    id='summary-order',
                    options=[
                        {'label': 'Order by Name', 'value': 'name'},
                        {'label': f'Order by Train {metric}', 'value': 'train_acc'},
                        {'label': f'Order by Test {metric}', 'value': 'test_acc'}
                    ],
                    value='name',
                    clearable=False,
                    style={'width': '400px'}
                )], style={"margin-left": "80px"})

            return ret

        @self.app.callback(
            Output("summary-graph", "children"),
            [Input("summary-order", "value")]
        )
        def graph(order):

            metric = 'RMSE' if self.train_page.ml_type == 'regression' else 'Accuracy'
            df_model = self.preprocess_data('Model', metric, order)
            labels_model = df_model['Model'] + " - " + df_model['Imputation']
            fig_model = self.create_fig(df_model, metric, labels_model)

            df_imp = self.preprocess_data('Imputation', metric, order)
            labels_imp = df_imp['Imputation'] + " - " + df_imp['Model']
            fig_imp = self.create_fig(df_imp, metric, labels_imp)

            x_title = "RMSE"
            if metric == "Accuracy":
                x_title = metric + " [%]"
                metric = metric.lower()
            fig_model.update_layout(
                title=f"Comparison of train and test {metric} by machine learning models, imputation techniques"
                      f" and their hyperparameters. Grouped by machine learning algorithms.",
                xaxis=dict(title=x_title)
            )
            fig_imp.update_layout(
                title=f"Comparison of train and test {metric} by machine learning models, imputation techniques"
                      f" and their hyperparameters. Grouped by imputation methods.",
                xaxis=dict(title=x_title)
            )

            return html.Div([dcc.Graph(figure=fig_model), dcc.Graph(figure=fig_imp)])

    def time_callback(self):
        @self.app.callback(
            Output("summary-time", "children"),
            Input("upload-flag", "data")
        )
        def time(_):
            if self.train_page.original_data is None or not self.train_page.imp_time["Imputation"]:
                return None

            df = pd.DataFrame(self.train_page.imp_time)
            df['Hyperparams'] = df['Hyperparams'].apply(str)
            df = df.drop_duplicates(subset=['Imputation', 'Hyperparams'], keep='last')
            df = df.sort_values(['Time'], ascending=False)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=df['Imputation'] + df['Hyperparams'],
                x=df['Time'],
                name='Time (seconds)',
                customdata=df['Hyperparams'],
                hovertemplate='<i>Hyperparams</i><br>%{customdata}<br>' +
                              '<br>' +
                              '<i>Time taken</i>: %{x}s<br>',
                marker=dict(color='#2e9cf4'),
                text=df['Time'],
                textposition='outside',
                orientation='h',  # make bars horizontal,
            ))

            x_labels = df['Imputation']
            graph_height = max(450, 50*len(df))
            fig.update_layout(
                title="Imputation time",
                barmode='group',
                yaxis=dict(
                    tickmode='array',
                    tickvals=list(range(len(x_labels))),
                    ticktext=x_labels
                ),
                xaxis=dict(  # larger xaxis because of text above bars
                    range=[0, max(df['Time']) * 1.2],
                    title="Time [s]"
                ),
                height=graph_height
            )

            return dcc.Graph(figure=fig)
