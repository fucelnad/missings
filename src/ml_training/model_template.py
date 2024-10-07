from abc import ABC, abstractmethod
import src.ml_training.trainer as trainer


class ModelTemplate(ABC):
    """A base class for ML models"""

    def __init__(self, imp_meth=None, imp_params=None, imp_time=None):

        self.fill_orig = True  # set to False if model can handle NaN
        self.model = None  # displayed in the summary graph as name of method
        self.model_imp = None  # class implementing ML model
        self.metric = None  # metric used for comparison of models
        self.metric_name = None  # metric name shown in the summary graph
        self.normalise = False  # set to True if data should be normalized before training

        self.imp_method = imp_meth
        self.imp_params = imp_params
        self.imp_time = imp_time

        self.target = None
        self.orig_acc = None
        self.imp_acc = None
        self.train_page = None

    @abstractmethod
    def train_model(self, train_page):
        """Handles training of the models on original and imputed data and their outputs.
        Updates trained_models with results achieved on original data, so it is not retrained each time.
        Updates results to train_page.used_algo, so they can be plotted in summary graph.
        Returns the output which is displayed in evaluation page."""

        pass

    def handle_models(self, train_page, target):
        """Handles how are the results showed on the page.
        Handles if both results on original and imputed dataset were given."""

        self.target = target
        self.train_page = train_page
        orig, imputed = self.train_model(train_page)

        if train_page.imputed_data is None:
            return trainer.output_one_model(orig)

        return trainer.output_two_models(orig, imputed)
