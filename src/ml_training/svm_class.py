from sklearn.metrics import accuracy_score

from sklearn import svm
from src.ml_training.class_template import ClassTemplate


class SVMClass(ClassTemplate):

    def __init__(self, imp_meth=None, imp_params=None, imp_time=None):

        super().__init__(imp_meth, imp_params, imp_time)
        self.model = "SVM"
        self.model_imp = svm.SVC()
        self.metric = accuracy_score
        self.metric_name = "Accuracy"
        self.fill_orig = True  # SVM can not handle NaNs
