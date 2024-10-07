import base64
import dash
import pandas as pd
import pytest
import sys
import os

app = dash.Dash(__name__, prevent_initial_callbacks='initial_duplicate')

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from src.pages.page_data import DataPage
from src.pages.page_impute import ImputePage
from src.pages.page_train import TrainPage
from src.pages.page_stats import StatsPage
import src.data_loading.data_handler as data_handler


@pytest.fixture
def page_data():
    instance = DataPage(app, False, None, None)
    data = {'Name': ['John', 'Anna', 'Peter', 'Linda'], 'Age': [28, 24, 35, 32]}
    instance.filename = "people.csv"
    instance.original_data = pd.DataFrame(data)
    instance.data = pd.DataFrame(data)

    instance.train_page = TrainPage(app)
    instance.impute_page = ImputePage(app, instance.train_page)
    instance.stats_page = StatsPage(app)
    return instance


def test_parse_contents(page_data):
    """Tests parsing of csv file.
    Fewer rows than 30 provided."""

    csv_data = "some,csv,data\n1,2,3"
    contents = "data:text/csv;base64," + base64.b64encode(csv_data.encode('utf-8')).decode('utf-8')
    filename = "data.csv"
    page_data.parse_contents(contents, filename)

    assert page_data.filename is None
    assert page_data.data is None


def test_invalid_data(page_data):
    """Tests invalid data upload"""

    contents = "no valid data"
    filename = "invalid_file"

    with pytest.raises(Exception):
        page_data.parse_contents(contents, filename)


def test_invalid_csv(page_data):
    """Tests invalid csv file upload"""

    contents = "no valid data"
    filename = "invalid_file.csv"

    with pytest.raises(Exception):
        page_data.parse_contents(contents, filename)


def test_reset_data(page_data):
    """Tests reset data"""

    page_data.target = "Name"
    page_data.problem = "regression"

    page_data.reset_data()
    assert page_data.data is None
    assert page_data.target is None
    assert page_data.problem is None


def test_filter_data(page_data):
    """Tests if data remains unchanged after filtering"""

    curr_data = page_data.data.copy(deep=True)
    data_handler.filter_data(page_data.data, 1, 5, "{Age} scontains 35")

    pd.testing.assert_frame_equal(page_data.data, curr_data)


def test_updating_data(page_data):
    """Tests if data are uploaded to other pages."""

    data = page_data.data
    page_data.update_pages(data, False)

    assert page_data.impute_page.original_data is None
    assert page_data.train_page.original_data is None

    page_data.update_pages(data, True)
    assert page_data.impute_page.original_data is not None
    assert page_data.train_page.original_data is not None
    assert page_data.impute_page.imputed_data is None
    assert page_data.train_page.imputed_data is None

    for df1, df2 in zip(page_data.impute_page.original_data, page_data.train_page.original_data):
        pd.testing.assert_frame_equal(df1, df2)


def test_handle_target(page_data):
    """Tests if modal is displayed correctly when target is selected."""

    df = pd.DataFrame({
        'target': [1, 2, None, 4, 5],
        'feature1': [6, 7, 8, 9, 10],
        'feature2': [11, 12, 13, 14, 15]
    })

    page_data.data = df
    page_data.target = "target"
    modal = page_data.handle_target()
    assert modal is not None

    df = pd.DataFrame({
        'target': [1, 2, 5, 4, 5],
        'feature1': [6, 7, 8, 9, 10],
        'feature2': [11, 12, 13, 14, 15]
    })

    page_data.data = df
    page_data.target = "target"
    modal = page_data.handle_target()
    assert modal is None

    df = pd.DataFrame({
        'target': [1, 2, 5, 4, 5, 1, 2, 5, 4, 5],
        'feature1': [None, None, None, None, None, None, 8, None, None, None],
        'feature2': [11, 12, 13, 14, 15, 11, 12, 13, 14, 15]
    })

    page_data.data = df
    page_data.target = "target"
    modal = page_data.handle_target()
    assert modal is not None


def test_update_options(page_data):
    """Tests updating of the selected options."""

    page_data.data = pd.DataFrame({'target': [1, 2, 3], 'col1': [4, 5, 6], 'col2': [7, 8, 9]})

    # no clicks
    assert page_data.update_options(None, 'target', 'problem', ['col1', 'col2']) == [dash.no_update] * 4
    assert page_data.update_options(0, 'target', 'problem', ['col1', 'col2']) == [dash.no_update] * 4

    # problem or target not provided
    assert page_data.update_options(1, None, 'problem', ['col1', 'col2'])[1] != ""
    assert page_data.update_options(1, 'target', None, ['col1', 'col2'])[1] != ""

    # target not numerical in regression problems
    page_data.data = pd.DataFrame({'target': ['a', 'b', 'c'], 'col1': [4, 5, 6], 'col2': [7, 8, 9]})
    assert page_data.update_options(1, 'target', 'regression', ['col1', 'col2'])[1] != ""

    # normal case
    page_data.data = pd.DataFrame({'target': [1, 2, 3], 'col1': [4, 5, 6], 'col2': [7, 8, 9]})
    assert page_data.update_options(1, 'target', 'classification', ['col1', 'col2'])[1] is None
