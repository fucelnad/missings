from sklearn.metrics import root_mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

from src.ml_training.regr_template import RegressionTemplate


class kNNRegressor(RegressionTemplate):

    def __init__(self, imp_meth=None, imp_params=None, imp_time=None):

        super().__init__(imp_meth, imp_params, imp_time)
        self.model = "kNN"
        self.model_imp = KNeighborsRegressor()
        self.metric = root_mean_squared_error
        self.metric_name = "RMSE"
        self.fill_orig = True  # kNN can not handle NaNs
        self.normalise = True
