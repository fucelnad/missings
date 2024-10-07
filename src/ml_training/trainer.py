from dash import dcc, html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import OneHotEncoder

from src.imputation.imputer import get_numeric_features
np.random.seed(42)


def conf_matrix_graph(true, predict, labels):
    """Creates graph of confusion matrix"""

    cm = confusion_matrix(true.astype(str), predict.astype(str), labels=labels, normalize="all")
    cm = np.around(cm*100, 2)
    fig = px.imshow(cm, text_auto=True,
                    labels=dict(x="Predicted labels", y="True labels", color="Percentage"),
                    x=labels,
                    y=labels,
                    color_continuous_scale=['#ffffff', '#52b5ff'], width=500, height=500,
                    zmin=0, zmax=100)

    # add lines
    for i in range(len(labels) + 1):
        fig.add_shape(type='line', x0=i - 0.5, x1=i - 0.5, y0=-0.5, y1=len(labels) - 0.5,
                      line=dict(color='black', width=2))

        fig.add_shape(type='line', x0=-0.5, x1=len(labels) - 0.5, y0=i - 0.5, y1=i - 0.5,
                      line=dict(color='black', width=2))

    fig.update_xaxes(side="top")
    return fig


def roc_curve_graph(true, score, num_classes, pos_label):
    """Returns ROC curve in case of binary classification with classes {0, 1} or {-1, 1}. None otherwise."""
    # Source:
    #   Title: ROC and PR Curves in Python
    #   Available at: https://plotly.com/python/roc-and-pr-curves/
    #   Accessed on: 01-04-2024

    if num_classes != 2 or (not set(true).issubset({0, 1}) and not set(true).issubset({-1, 1})):
        return None, None

    if np.ndim(score) != 1:
        score = score[:, 1]

    fpr, tpr, _ = roc_curve(true, score)

    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=500, height=500,
        color_discrete_sequence=['#2e9cf4']
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    return fig, pos_label


def hist_target_graph(orig, predict, fix_y):
    """Returns histogram of true and predicted numeric target values."""

    trace1 = go.Histogram(x=predict, opacity=0.75, name='Predicted values', nbinsx=50, marker={'color': "#2e9cf4"})
    trace2 = go.Histogram(x=orig, opacity=0.75, name='True values', nbinsx=50, marker={'color': "#bd5104"})
    trace3 = go.Histogram(x=fix_y, opacity=0, name='', nbinsx=50, hoverinfo="none")

    data = [trace1, trace2, trace3]

    layout = go.Layout(
        title="Histogram of target",
        xaxis=dict(title='Value'),
        yaxis=dict(title='Frequency'),
        barmode='overlay',
        showlegend=True
    )

    fig = go.Figure(data=data, layout=layout)
    return fig


def create_output_reg(train_mse, test_mse, fig):
    """Returns output of trained model."""

    return html.Div([
        html.H5(f'Train RMSE: {round(train_mse, 2)}', style={"margin": "10px"}),
        html.H5(f'Test RMSE: {round(test_mse, 2)}', style={"margin": "10px"}),
        html.Br(),
        html.H5("Target predictions (test subset)", style={"margin": "10px"}),
        dcc.Graph(figure=fig)
    ], style={"display": "flex", "flex-direction": "column", "justify-content": "center", "align-items": "center"})


def create_output_clf(train_acc, test_acc, fig, roc):
    """Returns output of trained model."""

    roc_graph = None
    if roc:
        roc_graph = dcc.Graph(figure=roc)

    return html.Div([
        html.H5(f'Train Accuracy: {round(train_acc, 2)}%', style={"margin": "10px"}),
        html.H5(f'Test Accuracy: {round(test_acc, 2)}%', style={"margin": "10px"}),
        html.Br(),
        html.H5("Target predictions (test subset)", style={"margin": "10px"}),
        dcc.Graph(figure=fig),
        roc_graph
    ], style={"display": "flex", "flex-direction": "column", "justify-content": "center", "align-items": "center"})


def output_one_model(model_output):
    """Creates output layout in case only one model was trained."""

    return html.Div([html.H4("Model without imputed values",
                             style={"margin": "10px", "display": "flex", "flex-direction": "column", "justify-content": "center", "align-items": "center"}),
                     model_output],
                    style={"border": "1px solid #ddd", "border-radius": "5px", "padding": "10px",
                           "background-color": "#f9f9f9", "margin": "10px"})


def output_two_models(orig, imputed):
    """Creates output layout in case two models were trained (original and imputed)."""

    return html.Div([
        html.Div([html.H4("Model without imputed values",
                          style={"margin": "10px", "display": "flex", "flex-direction": "column", "justify-content": "center", "align-items": "center"}),
                  orig],
                 style={"border": "1px solid #ddd", "border-radius": "5px", "padding": "10px",
                        "background-color": "#f9f9f9", "margin": "10px", "width": "50%", "float": "left"}),
        html.Div([html.H4("Model with imputed values",
                          style={"margin": "10px", "display": "flex", "flex-direction": "column", "justify-content": "center", "align-items": "center"}),
                  imputed],
                 style={"border": "1px solid #ddd", "border-radius": "5px", "padding": "10px",
                        "background-color": "#f9f9f9", "margin": "10px", "width": "50%", "float": "right"})
    ], style={"display": "flex"})


def update_summary(curr_dict, model, acc, imp_meth="Orig", imp_params={}, metric="Accuracy"):
    """Updates data used in summary page for graph generation."""

    curr_dict["Model"].append(model)
    curr_dict["Imputation"].append(imp_meth)
    curr_dict["Hyperparams"].append(imp_params.copy())
    curr_dict[f"Train {metric}"].append(round(acc[0], 2))
    curr_dict[f"Test {metric}"].append(round(acc[1], 2))


def impute_data(train, test):
    """In original dataset imputes missing values with -1 and adds a new binary feature telling if it was missing."""

    missing_cols = [col for col in train.columns if train[col].isnull().any()]
    num_cols, _ = get_numeric_features(train)
    for column in missing_cols:
        if train[column].isnull().any():
            train[column + '_was_missing'] = train[column].isnull()
            test[column + '_was_missing'] = test[column].isnull()

            fill_value = -1
            if column not in num_cols:
                fill_value = "-1"
            if train[column].dtype == "category":
                if fill_value not in train[column].cat.categories:
                    train[column] = train[column].cat.add_categories(fill_value)
                if fill_value not in test[column].cat.categories:
                    test[column] = test[column].cat.add_categories(fill_value)
            train[column].fillna(fill_value, inplace=True)
            test[column].fillna(fill_value, inplace=True)

    return train, test


def encode_data(X_train, X_test):
    """One hot encodes categorical columns in train and test set."""

    _, categorical_cols = get_numeric_features(X_train)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    X_train_encoded = pd.DataFrame(encoder.fit_transform(X_train[categorical_cols]))
    X_train_encoded.columns = encoder.get_feature_names_out(categorical_cols)
    X_train.drop(categorical_cols, axis=1, inplace=True)
    X_train.reset_index(drop=True, inplace=True)
    X_train = pd.concat([X_train, X_train_encoded], axis=1)

    X_test_encoded = pd.DataFrame(encoder.transform(X_test[categorical_cols]))
    X_test_encoded.columns = encoder.get_feature_names_out(categorical_cols)
    X_test.drop(categorical_cols, axis=1, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    X_test = pd.concat([X_test, X_test_encoded], axis=1)

    return X_train, X_test


def count_precision(method, train, test):
    """Counts precision by given method."""

    train = method(train[1], train[0])
    test = method(test[1], test[0])
    if method.__name__ == "accuracy_score":
        train *= 100
        test *= 100
    precision = (train, test)
    return precision


def update_train_orig(train_orig, model, acc, fig):

    train_orig[model] = {
        "Train Accuracy": acc[0],
        "Test Accuracy": acc[1],
        "Fig": fig
    }
