from sklearn.preprocessing import MinMaxScaler
import src.ml_training.trainer as trainer
import numpy as np

from src.ml_training.model_template import ModelTemplate


class ClassTemplate(ModelTemplate):

    def __init__(self, imp_meth=None, imp_params=None, imp_time=None):
        super().__init__(imp_meth, imp_params, imp_time)

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def prepare_data(self, fill_nan=False, imputed=False):
        """Prepares the data for training"""

        if imputed:
            train, test = self.train_page.imputed_data
        else:
            train, test = self.train_page.original_data
        if fill_nan:
            train, test = trainer.impute_data(train.copy(), test.copy())

        self.X_train = train.drop(self.target, axis=1)
        self.X_test = test.drop(self.target, axis=1)
        self.y_train = train[self.target]
        self.y_test = test[self.target]
        self.X_train, self.X_test = trainer.encode_data(self.X_train, self.X_test)

        if self.normalise:
            scaler = MinMaxScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)

        return self.X_train, self.y_train, self.X_test, self.y_test

    def predict(self, fill_nan, imputed=False):
        """Returns predictions on trained model."""

        X_train, y_train, X_test, y_test = self.prepare_data(fill_nan, imputed)
        clf = self.model_imp
        clf.fit(X_train, y_train)
        train_pred = clf.predict(X_train)
        test_pred = clf.predict(X_test)
        if hasattr(clf, 'decision_function'):
            score = clf.decision_function(X_test)
        else:
            score = clf.predict_proba(X_test)
        return train_pred, test_pred, score

    def store_results_orig(self, train_pred, test_pred, score):
        """Stores the results of the training"""

        train_orig = (train_pred, self.y_train)
        test_orig = (test_pred, self.y_test)
        acc_orig = trainer.count_precision(self.metric, train_orig, test_orig)
        used_algo = self.train_page.used_algo

        trainer.update_summary(used_algo, self.model, acc_orig, "Origin", {}, self.metric_name)
        return train_orig, test_orig, acc_orig, score

    def store_results_imp(self, train_pred, test_pred, score):
        """Stores the results of the training"""

        train_imp = (train_pred, self.y_train)
        test_imp = (test_pred, self.y_test)
        acc_imp = trainer.count_precision(self.metric, train_imp, test_imp)
        used_algo = self.train_page.used_algo

        trainer.update_summary(used_algo, self.model, acc_imp, self.train_page.imp_method,
                               self.train_page.imp_params, self.metric_name)
        return train_imp, test_imp, acc_imp, score

    def train_orig_data(self, fill_nan=False):
        """Returns predictions on train and test set of original data together with accuracy"""

        trained_models = self.train_page.train_orig
        if self.model not in trained_models:
            train_pred, test_pred, orig_score = self.predict(fill_nan)
            train_orig, test_orig, acc_orig, score = self.store_results_orig(train_pred, test_pred, orig_score)
            trained_models[self.model] = \
                {self.metric_name: acc_orig, "Predictions": (train_orig, test_orig), "Score": score}
            return train_orig, test_orig, acc_orig, score
        else:
            train_orig, test_orig = trained_models[self.model]["Predictions"]
            acc_orig = trained_models[self.model][self.metric_name]
            orig_score = trained_models[self.model]["Score"]

        return train_orig, test_orig, acc_orig, orig_score

    def train_imp_data(self):
        """Returns predictions on train and test set of imputed data together with accuracy"""

        train_pred, test_pred, imp_score = self.predict(False, True)
        return self.store_results_imp(train_pred, test_pred, imp_score)

    def train_model(self, train_page):
        """Trains model for classification and returns its visualization and results."""

        train_orig, test_orig, acc_orig, orig_score = self.train_orig_data(self.fill_orig)
        num_classes = train_orig[1].nunique()
        acc_orig = (acc_orig[0], acc_orig[1])

        labels_orig = np.unique(np.concatenate((test_orig[1], train_orig[1]))).astype(str)
        fig_orig = trainer.conf_matrix_graph(test_orig[1], test_orig[0], labels_orig)
        roc_orig, pos_label = trainer.roc_curve_graph(test_orig[1], orig_score, num_classes, None)

        if train_page.imputed_data is None:
            output_orig = trainer.create_output_clf(acc_orig[0], acc_orig[1], fig_orig, roc_orig)
            return output_orig, None

        train_imp, test_imp, acc_imp, imp_score = self.train_imp_data()
        acc_imp = (acc_imp[0], acc_imp[1])
        labels_imp = np.unique(np.concatenate((test_imp[1], train_imp[1]))).astype(str)
        fig_imp = trainer.conf_matrix_graph(test_imp[1], test_imp[0], labels_imp)
        roc_imp, _ = trainer.roc_curve_graph(test_imp[1], imp_score, num_classes, pos_label)

        output_orig = trainer.create_output_clf(acc_orig[0], acc_orig[1], fig_orig, roc_orig)
        output_imp = trainer.create_output_clf(acc_imp[0], acc_imp[1], fig_imp, roc_imp)

        return output_orig, output_imp
