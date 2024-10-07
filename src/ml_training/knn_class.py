from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from src.ml_training.class_template import ClassTemplate


class kNNClass(ClassTemplate):

    def __init__(self, imp_meth=None, imp_params=None, imp_time=None):

        super().__init__(imp_meth, imp_params, imp_time)
        self.model = "kNN"
        self.model_imp = KNeighborsClassifier()
        self.metric = accuracy_score
        self.metric_name = "Accuracy"
        self.fill_orig = True  # kNN can not handle NaNs
        self.normalise = True
