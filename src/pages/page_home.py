import base64
from dash import html
import dash_bootstrap_components as dbc
import urllib

from src.pages.page_template import Page


class HomePage(Page):
    """Class representing home page with basic information about the app"""
    def create_basic_layout(self):
        info = html.P("Missing values are present in most of the real-world datasets. They occur for various "
                      "reasons that can not always be eliminated. Therefore, it is crucial to know how to deal with "
                      "them and what the options are. This application focuses only on the imputation of data. You can "
                      "upload a dataset and choose one of the five imputation techniques offered: mean imputation, "
                      "kNN, MissForest, MICE and GAIN. You can also select some of the hyperparameters. After the "
                      "imputation, a graphical representation of the results is generated. You can also train some of "
                      "the machine learning models on the data and compare how it has affected the final results. "
                      "Hyperparameter random_state controlling randomness is set to 42. This application consists of several pages. To get the most out of "
                      "the application, please follow the diagram below.",
                      style={"textAlign": "justify"})
        encoded_image = base64.b64encode(open("src/program_files/diagram.png", "rb").read())
        encoded_image = "data:image/png;base64," + urllib.parse.quote(encoded_image)
        diagram = html.Img(src=encoded_image, style={"height": "100%", "width": "100%"})

        return html.Div([
            html.H3("INTRODUCTION", style={"margin-top": "20px"}),
            html.Br(),
            info,
            html.Br(),
            diagram
        ], style={"display": "flex", "flex-direction": "column", "align-items": "center", "justify-content": "center",
                  "margin-left": "100px", "margin-right": "100px"})
