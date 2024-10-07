import ast
import dash
from dash import dash_table, dcc, html
import dash_bootstrap_components as dbc
import math
import pandas as pd
from sklearn.model_selection import train_test_split

OPERATORS = [["ge ", ">="], ["le ", "<="], ["lt ", "<"], ["gt ", ">"], ["ne ", "!="], ["eq ", "="], ["contains "]]
MAX_CATEGORY = 10  # sets maximum number of unique values in numerical feature to be cast to categorical


def upload_button(size):
    """Helper function to create upload button"""

    button = dcc.Upload(id="upload-data", children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
                        style={"width": "87%", "height": "60px", "lineHeight": "60px", "borderWidth": "1px",
                               "borderStyle": "dashed", "borderRadius": "5px", "textAlign": "center",
                               "margin-left": "100px", "margin-right": "100px"},
                        multiple=True, max_size=100**8)

    text = html.P(f"Upload the dataset here. Only CSV files with size smaller than {size} MB will be accepted. You "
                  f"can customise it by deleting some of the features or by changing their data types. To move on to the next pages"
                  f" select the target that will be predicted by machine learning models and the type of problem."
                  f" At this point the dataset is split into training and test sets, with the training set making up 80 % of the original dataset."
                  f" Note that missing values are only considered as missing if not present (e.g. they are NOT denoted "
                  f" by -, ?, None, ...). You can also search for missing values in the features of the uploaded dataset by"
                  f" typing \"NaN\" in the search box. Additionaly, you can use the search boxes for filtering on values.",
                  style={"textAlign": "justify"})

    return html.Div([
        html.Div([html.H3("UPLOAD DATA", style={"margin-top": "20px"}),
                  html.Br(),
                  html.P(text)],
                 style={"display": "flex", "flex-direction": "column", "align-items": "center",
                        "justify-content": "center", "margin-left": "100px", "margin-right": "100px"}),
        button
        ])


def select_target(data):
    """Helper function to create target dropdown and choose type of ML problem."""

    if data is None:
        return []

    available_columns = [col for col in data.columns if data[col].notna().any()]  # not completely missing columns

    return html.Div([
        html.H4("Pick a target variable from uploaded dataset", style={"margin": "10px"}),
        dcc.Dropdown(
            id="target-dropdown",
            options=[{"label": col, "value": col} for col in available_columns],
            placeholder="Select a target",
            style={"margin": "10px", "width": "600px"}
        ),
        html.Br(),
        html.H4("Pick a type of machine learning problem", style={"margin": "10px"}),
        dcc.RadioItems(
            id='problem-button',
            options=[
                {'label': 'Classification', 'value': 'classification'},
                {'label': 'Regression', 'value': 'regression'}
            ],
            style={"margin": "10px"}
        ),
        html.Button('Confirm', id='confirm-button', style={"margin": "10px"})
    ], style={"border": "1px solid #ddd", "border-radius": "5px", "padding": "10px",
              "background-color": "#f9f9f9", "margin": "10px", 'minHeight': '200px'}
    )


def show_data(df_columns):
    """Creates layout for data preview"""

    return html.Div(
        className="scrollable-content",
        children=[html.Div([
            dash_table.DataTable(
                id="datatable-paging",
                columns=[{"name": i, "id": i} for i in df_columns],
                page_current=0,
                page_size=10,
                page_action="custom",
                style_cell={"minWidth": "140px", "maxWidth": "200px", "overflow": "auto"},
                style_table={"overflowX": "auto"},
                filter_action="custom",
                filter_query=""
            )
        ], style={"margin": "10px"})]
    )


def data_preview(data, filename, target=None, problem=None):
    """Helper function to preview the data"""

    text = html.Div([html.H5("Preview data: " + filename, style={"margin": "10px"}),
                     html.H5("Chosen target: " + str(target), style={"margin": "10px"}),
                     html.H5("Chosen problem type: " + str(problem), style={"margin": "10px"})])

    dropdown = dbc.Row([
        dbc.Col([
            html.H5("Pick columns to delete", style={"margin": "10px"}),
            dcc.Dropdown(
                id='data-columns',
                options=[{'label': i, 'value': i} for i in data.columns],
                multi=True,
                placeholder="Select columns",
                style={"margin-top": "10px", "margin-left": "5px", "width": "630px"}
            ),
            html.Button("Confirm", id="confirm-del", style={"margin-left": "10px", "margin-top": "5px"})
        ], width=9)
    ])

    layout = dbc.Row([
        dbc.Col(text, width=6),
        dbc.Col(dropdown, width=6)
    ])

    table = show_data(data.columns)

    return html.Div([layout, table], style={"overflowX": "hidden"})


def display_error_message(err):
    """Displays an error message"""

    return html.Div([html.H3(err)], style={"margin-left": "50px"})


def split_filter_part(filter_part):
    """Filters records in the data table"""
    # Source:
    # Title: DataTable - Python Callbacks
    # Available at: https://dash.plotly.com/datatable/callbacks
    # Accessed on: 07-02-2024

    for operator_type in OPERATORS:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find("{") + 1: name_part.rfind("}")]

                value_part = value_part.strip()
                v0 = value_part[0]
                if v0 == value_part[-1] and v0 in (""", """, "`"):
                    value = value_part[1: -1].replace("\\" + v0, v0)
                else:
                    try:
                        if value_part == "NaN":
                            raise ValueError
                        value = float(value_part)
                    except ValueError:
                        value = value_part

                return name, operator_type[0].strip(), value

    # no operator found
    return [None] * 3


def split_data(data):
    """Splits data to train and test subsets, where test subset is 20%"""

    if data is None:
        return None

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    return [train_data, test_data]


def filter_data(data, page_current, page_size, filter_used):
    """Handles filtering and scrolling in data preview"""
    # Source:
    # Title: DataTable - Python Callbacks
    # Available at: https://dash.plotly.com/datatable/callbacks
    # Accessed on: 07-02-2024

    filtering_expressions = filter_used.split(" && ")
    dff = data
    for filter_part in filtering_expressions:
        col_name, operator, filter_value = split_filter_part(filter_part)

        try:
            if operator in ("eq", "ne", "lt", "le", "gt", "ge"):
                dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
            elif operator == "contains":
                if filter_value == "NaN":
                    dff = dff[dff[col_name].isna()]
                elif col_name in dff.columns and pd.api.types.is_string_dtype(dff[col_name]):
                    dff = dff.dropna(subset=[col_name])
                    dff = dff.loc[dff[col_name].str.contains(filter_value)]
                else:
                    dff = dff.loc[getattr(dff[col_name], "eq")(filter_value)]
            elif operator is not None:
                dff = pd.DataFrame(columns=dff.columns)
        except Exception as e:
            print(f"Raised exception in filtering: {e}")
            dff = pd.DataFrame(columns=dff.columns)

    num_pages = math.ceil(dff.shape[0] / 10)
    return num_pages, dff.iloc[page_current * page_size:(page_current + 1) * page_size].to_dict("records")


def problem_type():
    """Creates radio button to pick classification or regression problem"""

    return html.Div([
        dcc.RadioItems(
            id='problem-button',
            options=[
                {'label': 'Classification', 'value': 'classification'},
                {'label': 'Regression', 'value': 'regression'}
            ],
            value='classification'
        )])


def select_data_types(data, int64_cols):
    """Provides dropdown with data types for each column of dataset."""

    if data is not None:
        components = []
        dropdown_components = []
        components.append(html.H4("Pick column data types", style={"margin": "10px"}))

        for col in data.columns:
            col_type = str(data[col].dtype)
            if col in int64_cols:
                col_type = "int64"
            dropdown = dcc.Dropdown(
                id={'type': 'setting', 'index': col},
                options=[
                    {'label': 'Int64', 'value': 'int64'},
                    {'label': 'Float64', 'value': 'float64'},
                    {'label': 'Category', 'value': 'category'}
                ],
                value=col_type,
                clearable=False
            )

            dropdown_components.append(html.Div([
                html.Div(f'{col}',
                         style={"margin": "10px", "overflow": "auto", "max-width": "400px", "width": "70%"}),
                html.Div(dropdown, style={"margin": "10px", "width": "30%"})
            ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'space-between'}))

        components.append(
            html.Div(dropdown_components,
                     style={'overflowY': 'auto', 'overflowX': 'hidden', 'height': '238px'})
        )

        return html.Div(components, style={"border": "1px solid #ddd", "border-radius": "5px", "padding": "10px",
                                           "background-color": "#f9f9f9", "margin": "10px"})


def handle_types_choice(data_types, data, ctx, int64_cols):
    """Handles the casting of data columns to other types.
    If possible the selected column is cast, otherwise warn message it displayed and the type stays unchanged."""

    if not ctx.triggered:
        return [dash.no_update] * 4, False
    int64_cols = set(int64_cols)
    prop = ctx.triggered[0]['prop_id']
    col = ast.literal_eval(prop.rsplit('.', 1)[0])['index']
    pos = list(data.columns).index(col)
    if data[col].dtype == data_types[pos] and col not in int64_cols:
        return [dash.no_update] * 4, False

    try:
        # cast numeric column to category only if less than 10 unique values
        if data_types[pos] == "category" and len(data[col].unique()) > MAX_CATEGORY and data[col].dtype != "category":
            data[col] = data[col].astype(data[col].dtype)
            modal = create_warn_modal(f"Column {col} can not be cast to {data_types[pos]} due to too many unique "
                                      f"values exceeding threshold {MAX_CATEGORY}.", "data-type")
            return [dash.no_update, modal, select_data_types(data, int64_cols), dash.no_update], False

        if data_types[pos] == "int64":
            curr_type = data[col].dtype
            data[col] = data[col].astype('float64')
            if not all(x.is_integer() for x in data[col].dropna()):
                data[col] = data[col].astype(curr_type)
                raise ValueError
            int64_cols.add(col)

        else:
            if col in int64_cols:
                int64_cols.remove(col)
            data[col] = data[col].astype(data_types[pos])
    except (ValueError, AttributeError):
        modal = create_warn_modal(f"Column {col} can not be cast to {data_types[pos]}.", "data-type")
        return [dash.no_update, modal, select_data_types(data, int64_cols), dash.no_update], False

    int64_cols = list(int64_cols)
    return ["change", dash.no_update, dash.no_update, int64_cols], True


def create_warn_modal(message, modal_id):
    """Creates modal pop-up window."""

    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Warning")),
        dbc.ModalBody(message)],
        id=modal_id,
        is_open=True,
        size="lg",
    )
