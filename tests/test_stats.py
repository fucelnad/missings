import pandas as pd

from src.stats import stats_generator


def test_is_numerical():
    """Tests if given column is numerical"""

    data = {
        'int_col': [1, 2, 3, 4, 5],
        'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
        'str_col': ['A', 'B', 'C', 'D', 'E']
    }
    df = pd.DataFrame(data)

    assert stats_generator.is_numerical(df, 'int_col')
    assert stats_generator.is_numerical(df, 'float_col')
    assert not stats_generator.is_numerical(df, 'str_col')


def test_graph():
    """Tests if graph was returned for given column name"""

    df = pd.DataFrame({'numbers': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]})
    col = 'numbers'

    graph = stats_generator.col_graph(df, col)
    assert graph is not None
