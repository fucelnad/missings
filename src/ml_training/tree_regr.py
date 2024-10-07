from sklearn.metrics import root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from src.ml_training.regr_template import RegressionTemplate


class TreeRegressor(RegressionTemplate):

    def __init__(self, imp_meth=None, imp_params=None, imp_time=None):

        super().__init__(imp_meth, imp_params, imp_time)
        self.model = "Decision Tree"
        self.model_imp = DecisionTreeRegressor(max_depth=10, random_state=42)
        self.metric = root_mean_squared_error
        self.metric_name = "RMSE"
        self.fill_orig = False  # Decision Tree can handle NaNs
