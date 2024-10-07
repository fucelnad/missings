from sklearn.preprocessing import MinMaxScaler
import src.ml_training.trainer as trainer

from src.ml_training.model_template import ModelTemplate


class RegressionTemplate(ModelTemplate):

    def __init__(self, imp_meth=None, imp_params=None, imp_time=None):
        super().__init__(imp_meth, imp_params, imp_time)
        self.fill_orig = True

    def get_results(self, data, target, model, fill_nan=False):
        """Function to train ML model. Returns predictions on train and test sets with true values."""

        train, test = data
        if fill_nan:
            train, test = trainer.impute_data(train.copy(), test.copy())

        X_train = train.drop(target, axis=1)
        y_train = train[target]
        X_test = test.drop(target, axis=1)
        y_test = test[target]
        X_train, X_test = trainer.encode_data(X_train, X_test)

        if self.normalise:
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        clf = model
        clf.fit(X_train, y_train)
        train_pred = clf.predict(X_train)
        test_pred = clf.predict(X_test)

        return (train_pred, y_train), (test_pred, y_test)

    def train_orig_data(self, fill_nan=False):
        """Returns predictions on train and test set of original data together with accuracy"""

        trained_models = self.train_page.train_orig
        used_algo = self.train_page.used_algo

        if self.model not in trained_models:
            train_orig, test_orig = self.get_results(self.train_page.original_data, self.target, self.model_imp,
                                                        fill_nan)
            acc_orig = trainer.count_precision(self.metric, train_orig, test_orig)
            # store model trained on original data
            trained_models[self.model] = {self.metric_name: acc_orig, "Predictions": (train_orig, test_orig)}
            trainer.update_summary(used_algo, self.model, acc_orig, "Origin", {}, self.metric_name)
        else:
            train_orig, test_orig = trained_models[self.model]["Predictions"]
            acc_orig = trained_models[self.model][self.metric_name]

        return train_orig, test_orig, acc_orig

    def train_imp_data(self):
        """Returns predictions on train and test set of imputed data together with accuracy"""

        used_algo = self.train_page.used_algo
        train_imp, test_imp = self.get_results(self.train_page.imputed_data, self.target, self.model_imp)
        acc_imp = trainer.count_precision(self.metric, train_imp, test_imp)
        trainer.update_summary(used_algo, self.model, acc_imp, self.train_page.imp_method, self.train_page.imp_params,
                               self.metric_name)

        return train_imp, test_imp, acc_imp

    def train_model(self, train_page):
        """Trains model for classification and returns its visualization and results."""

        train_orig, test_orig, acc_orig = self.train_orig_data(self.fill_orig)
        fig_orig = trainer.hist_target_graph(test_orig[1], test_orig[0], test_orig[0])
        if train_page.imputed_data is None:
            output_orig = trainer.create_output_reg(acc_orig[0], acc_orig[1], fig_orig)
            return output_orig, None

        train_imp, test_imp, acc_imp = self.train_imp_data()
        fig_imp = trainer.hist_target_graph(test_imp[1], test_imp[0], test_orig[0])

        output_orig = trainer.create_output_reg(acc_orig[0], acc_orig[1], fig_orig)
        output_imp = trainer.create_output_reg(acc_imp[0], acc_imp[1], fig_imp)

        return output_orig, output_imp
