import base64
from dash import dcc, html, dash_table
import matplotlib
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import urllib
import uuid

from src.imputation.imputer import get_numeric_features

matplotlib.use('Agg')

MAX_SAMPLE = 1500


def is_numerical(df, column):
    """Checks if given column is numerical"""

    return pd.api.types.is_numeric_dtype(df[column])


def corr_matrix(data):
    """Returns correlation matrix of columns in data"""

    numeric, _ = get_numeric_features(data)
    data = data[numeric]
    corr = data.corr()
    corr = corr.dropna(how='all').dropna(axis=1, how='all')

    heatmap = go.Heatmap(z=np.round(corr.values, 2),
                         x=corr.columns,
                         y=corr.columns,
                         colorscale='RdBu_r',
                         zmin=-1,
                         zmax=1,
                         hovertemplate='column 1: %{x}<br>column 2: %{y}<br>correlation: %{z}<extra></extra>')

    layout = go.Layout(title='Correlation matrix',
                       width=500, height=500)
    fig = go.Figure(data=[heatmap], layout=layout)

    return dcc.Graph(figure=fig)


def column_button(data, target):
    """Created dropdown button with names of columns in the dataset"""

    if target is None:
        target = data.columns.tolist()[0]

    return html.Div([
        html.Br(),
        html.H4("Statistics of columns", style={"margin": "10px"}),
        dcc.Dropdown(
            id="column-dropdown",
            options=[
                {"label": col, "value": col}
                for col in data.columns.tolist()
            ],
            value=target,
            placeholder="Select a column",
            style={"margin": "10px"}
        )
    ])


def missing_table(data, test):
    """Creates a dataset with the number and percentage of missing values in columns."""

    missing_values = data.isnull().sum()
    total_values = data.shape[0]
    percentage_train = round((missing_values / total_values) * 100, 2)

    missing_values = test.isnull().sum()
    total_values = test.shape[0]
    percentage_test = round((missing_values / total_values) * 100, 2)

    missing_data = pd.DataFrame({
        "Column": missing_values.index,
        "Train Set": percentage_train.values,
        "Test Set": percentage_test.values
    })

    missing_data = missing_data.sort_values(by="Train Set", ascending=False)

    output = html.Div([
        html.H4("Missing Data Percentage", style={"margin-left": "50px"}),
        dash_table.DataTable(
            id="missing-data-table",
            columns=[{"name": col, "id": col} for col in missing_data.columns],
            data=missing_data.to_dict("records"),
            style_cell={"minWidth": "140px", "maxWidth": "200px", "overflow": "auto"},
            style_table={"overflowY": "scroll", "maxHeight": "450px", "width": "100%", "marginLeft": "20px",
                         "marginRight": "auto", "width": "710px"}
        )
    ])
    return output


def missing_graph(data):
    """Makes matrix plot of missing values in random sample of dataset."""

    sample_size = min(MAX_SAMPLE, data.shape[0])
    file_name = uuid.uuid4()
    msno.matrix(data.sample(sample_size), color=(0.18, 0.61, 0.96))  # (46, 156, 244)
    plt.savefig(f"src/program_files/{file_name}_matrix.png", bbox_inches="tight")
    encoded_image = base64.b64encode(open(f"src/program_files/{file_name}_matrix.png", "rb").read())
    encoded_image = "data:image/png;base64," + urllib.parse.quote(encoded_image)
    matrix = html.Img(src=encoded_image, style={"height": "80%", "width": "80%", "display": "block",
                                                "margin-left": "auto", "margin-right": "auto"})

    msno.dendrogram(data)
    plt.savefig(f"src/program_files/{file_name}_dendrogram.png", bbox_inches="tight")
    encoded_image = base64.b64encode(open(f"src/program_files/{file_name}_dendrogram.png", "rb").read())
    encoded_image = "data:image/png;base64," + urllib.parse.quote(encoded_image)
    dendrogram = html.Img(src=encoded_image, style={"height": "80%", "width": "80%", "display": "block",
                                                    "margin-left": "auto", "margin-right": "auto"})

    try:
        os.remove(f"src/program_files/{file_name}_matrix.png")
        os.remove(f"src/program_files/{file_name}_dendrogram.png")
    except OSError as e:
        print("Warning: could not delete files")

    return html.Div([
        html.Br(),
        html.H4("Dendrogram of missing values", style={"margin": "10px"}),
        html.P([
            """The dendrogram is a tree diagram that groups features from the dataset using hierarchical clustering 
            algorithm. It works on the principle of mutual similarity, here determined by a nullity correlation. The 
            nullity correlation is defined on interval [-1, 1], where -1 means that if one feature is observed the other
            will certainly be missing. Value 0 means that the missingness of one feature says nothing about 
            the missingness of other feature. A value of 1 means that if one feature is observed the other will 
            certainly be observed too. [""",
            html.A("1", href="https://github.com/ResidentMario/missingno", target="_blank"),
            "]"
        ], style={"margin-left": "100px", "margin-right": "100px", "textAlign": "justify"}),
        html.P(["""The dendrogram is read from the top to the bottom. Features that are linked together at a distance of 0 fully
         predict each other's presence. Features that are split close to 0 predict each other's presence quite accurately, 
         but not perfectly. The height of the cluster on the left tells how many values would have to be either filled or 
         dropped for their nullity to correspond. [""",
         html.A("1", href="https://github.com/ResidentMario/missingno", target="_blank"),
         "]"
        ], style={"margin-left": "100px", "margin-right": "100px", "textAlign": "justify"}),
        dendrogram,
        html.H4("Matrix plot of missing values", style={"margin": "10px"}),
        html.P(
            [f"The matrix plot shows all data points by rectangles. If there are more data points than {MAX_SAMPLE}, "
             f"only randomly selected data points of total size {MAX_SAMPLE} are displayed. Blue colour means the data "
             f"point is observed, white means it is missing. This nullity matrix is a great tool for spotting"
             f" patterns in data completion. The graph is accompanied by a line plot on the right. It shows the"
             f" general shape of the completeness and highlights the rows with maximum and minimum nullity. [",
             html.A("1", href="https://github.com/ResidentMario/missingno", target="_blank"),
             ", ",
             html.A("2", href="https://link.springer.com/article/10.1007/s11634-011-0102-y", target="_blank"),
             "]"
             ], style={"margin-left": "100px", "margin-right": "100px", "textAlign": "justify"}),
        matrix,
    ], style={"text-align": "center"})


def col_graph(data, column):
    """
    Generates graphs based on the data.
    If there are more than 15 non-numerical values only number of unique values is returned.
    If there are more than 15 numerical values histogram is returned.
    Otherwise, bar plot is returned.
    """

    if column not in data.columns:
        return html.H5(f"",
                       style={"margin": "10px"})

    unique_values = data[column].nunique()

    if is_numerical(data, column):
        histogram = px.histogram(data, x=column, title=f"Histogram of \"{column}\"",
                                 color_discrete_sequence=['#2e9cf4'])
        histogram.update_layout(yaxis_title="Frequency")
        return dcc.Graph(figure=histogram)
    elif unique_values <= 15:
        value_counts = data[column].value_counts().reset_index()
        value_counts.columns = [column, "Count"]
        bar = px.bar(value_counts, x=column, y="Count", title=f"Bar plot of \"{column}\"",
                     color_discrete_sequence=['#2e9cf4'])
        bar.update_xaxes(tickvals=value_counts[column].unique())
        return dcc.Graph(figure=bar)
    else:
        return html.H5(f"The number of categories exceeds the threshold 25 and therefore the graph is not shown.",
                       style={"margin": "10px"})


def col_stats(data, column):
    """Displays statistic for column in the given data"""

    if column not in data.columns:
        return html.H5(f"Invalid column given. Please refresh the page to see the current data.",
                       style={"margin": "10px"})
    unique = data[column].value_counts()
    num_unique = data[column].nunique()

    stats = html.Div([
        html.Div([
            html.H5(f"Statistics for '{column}'", style={"margin": "10px", "max-width": "800px", "overflow": "auto"}),
            html.P(f"Unique values: {num_unique}", style={"margin": "10px"})
        ], style={"text-align": "left"})
    ], style={"height": "400px", "overflowY": "scroll", "border": "1px solid #ddd", "border-radius": "5px",
              "padding": "10px", "background-color": "#f9f9f9", "margin": "10px", "display": "flex"})

    if is_numerical(data, column):
        stats.children[0].children.extend([
            html.P(f"Minimum: {data[column].min():.2f}", style={"margin": "10px"}),
            html.P(f"Maximum: {data[column].max():.2f}", style={"margin": "10px"}),
            html.P(f"Mean: {data[column].mean():.2f}", style={"margin": "10px"}),
            html.P(f"Median: {data[column].median():.2f}", style={"margin": "10px"}),
            html.P(f"Standard deviation: {data[column].std():.2f}", style={"margin": "10px"})
        ])
    else:
        stats.children[0].children.extend([
            html.P(f"Mode: {data[column].mode()[0]}", style={"margin": "10px"})
        ])

    threshold = 25
    if num_unique <= threshold:
        stats.children[0].children.extend([html.Div([
            html.Br(),
            html.H5("Unique values", style={"margin": "10px"})
        ])])

        categories = html.Div([])
        for category, count in unique.items():
            categories.children.append(html.P(f"{category}: {count}", style={"margin": "10px"}))

        categories = html.Div(categories)
        stats.children[0].children.append(categories)
    return stats
