from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from src.ml_training.class_template import ClassTemplate


class TreeClass(ClassTemplate):

    def __init__(self, imp_meth=None, imp_params=None, imp_time=None):

        super().__init__(imp_meth, imp_params, imp_time)
        self.model = "Decision Tree"
        self.model_imp = DecisionTreeClassifier(max_depth=10, random_state=42)
        self.metric = accuracy_score
        self.metric_name = "Accuracy"
        self.fill_orig = False  # decision tree can handle NaNs
