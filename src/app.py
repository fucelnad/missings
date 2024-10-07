import dash
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from src.app_layout import AppLayout
from waitress import serve


# Roboto font
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MATERIA], suppress_callback_exceptions=True,
                prevent_initial_callbacks='initial_duplicate')
app_layout = AppLayout(app)

page_layouts = {
    "/": app_layout.home_page.layouts["basic"],
    "/data": app_layout.data_page.layouts["basic"],
    "/stats": app_layout.stats_page.layouts["basic"],
    "/impute": app_layout.impute_page.layouts["basic"],
    "/results": app_layout.res_page.layouts["basic"],
    "/train": app_layout.train_page.layouts["basic"],
    "/summary": app_layout.summary_page.layouts["basic"]
}


def display_page(pathname):
    """Page routing function"""

    return page_layouts.get(pathname, html.H5("Error: page not found.", style={"margin": "10px"}))


@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page_callback(pathname):
    """Callback to update displayed page based on pathname"""

    return display_page(pathname)


if __name__ == "__main__":
    serve(app.server, host="0.0.0.0", port=8080)
