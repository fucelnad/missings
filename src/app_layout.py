from dash import dcc, html
import dash_bootstrap_components as dbc

from src.pages.page_data import DataPage
from src.pages.page_home import HomePage
from src.pages.page_impute import ImputePage
from src.pages.page_results import ResultsPage
from src.pages.page_stats import StatsPage
from src.pages.page_summary import SummaryPage
from src.pages.page_train import TrainPage


class AppLayout:
    """Class representing the whole application"""

    def __init__(self, app):
        self.app = app
        self.app.layout = self.main_layout()

        self.home_page = HomePage(app)
        self.stats_page = StatsPage(app)
        self.train_page = TrainPage(app)
        self.summary_page = SummaryPage(app, self.train_page)
        self.impute_page = ImputePage(app, self.train_page)
        self.res_page = ResultsPage(app, self.impute_page)
        self.data_page = DataPage(app, self.stats_page, self.impute_page, self.train_page)

    def main_layout(self):
        """Creates the main layout of the page"""

        navbar = dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("UPLOAD DATA", href="data")),
                dbc.NavbarBrand("|", className="mx-2"),
                dbc.NavItem(dbc.NavLink("STATISTICS", href="stats")),
                dbc.NavbarBrand("|", className="mx-2"),
                dbc.NavItem(dbc.NavLink("IMPUTATION", href="impute")),
                dbc.NavbarBrand("|", className="mx-2"),
                dbc.NavItem(dbc.NavLink("RESULTS OF IMPUTATION", href="results")),
                dbc.NavbarBrand("|", className="mx-2"),
                dbc.NavItem(dbc.NavLink("EVALUATION", href="train")),
                dbc.NavbarBrand("|", className="mx-2"),
                dbc.NavItem(dbc.NavLink("SUMMARY", href="summary"))
            ],
            brand="VISUALIZATION APP",
            brand_href="/",
            color="primary",
            dark=True,
            fluid=True,
            links_left=True,
            sticky="top"
        )

        main_container = html.Div(
            [
                dcc.Location(id="url", refresh=False),
                dcc.Store(id="upload-flag", storage_type="memory"),  # flag for change in uploaded data
                dcc.Store(id="imputation-params", storage_type="memory"),  # flag for hyperparams tuning
                dcc.Store(id="imputation-time", storage_type="memory"),  # time that last imputation took
                dcc.Store(id="data-target", storage_type="memory"),  # currently chosen target
                dcc.Store(id="ml-type", storage_type="memory"),  # classification/regression problem
                dcc.Store(id="int64-cols", storage_type="session", data=[]),  # stores int64 columns
                dcc.Store(id="categorical-cols", storage_type="memory"),
                navbar,
                html.Div(
                    id="page-content",
                    children=[],
                    className="page-container",  # add className for styling
                ),
            ],
            className="main-container"  # add className for styling
        )

        return main_container
