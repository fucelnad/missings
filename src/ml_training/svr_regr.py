from sklearn.metrics import root_mean_squared_error
from sklearn import svm

from src.ml_training.regr_template import RegressionTemplate


class SVMRegressor(RegressionTemplate):

    def __init__(self, imp_meth=None, imp_params=None, imp_time=None):

        super().__init__(imp_meth, imp_params, imp_time)
        self.model = "SVM"
        self.model_imp = svm.SVR()
        self.metric = root_mean_squared_error
        self.metric_name = "RMSE"
        self.fill_orig = True  # SVM can not handle NaNs
