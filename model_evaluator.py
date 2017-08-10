import numpy as np
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib


class ModelEvaluator():

    def __init__(self, model, train, label, default_model_params={}):
        self.model = model
        self.train = train
        self.label = label
        self.default_model_params = default_model_params

    def cross_validate(self, scoring="neg_mean_squared_error", **kwargs):
        m = self.model(**self.default_model_params)
        self.scores = cross_val_score(
            m, self.train, y=self.label, scoring=scoring, **kwargs)
        self.rmse_scores = np.sqrt(-self.scores)

    def perform_search(self, search_method, param_grid,
                       scoring='neg_mean_squared_error', **kwargs):
        m = self.model(**self.default_model_params)
        self.search_cv = search_method(
            m, param_grid, scoring=scoring, **kwargs)
        self.search_cv.fit(self.train, self.label)

    def grid_search(self, *args, **kwargs):
        self.perform_search(GridSearchCV, *args, **kwargs)

    def random_search(self, *args, **kwargs):
        self.perform_search(RandomizedSearchCV, *args, **kwargs)

    def evaluate_best_estimator(self, train, label):
        predictions = self.search_cv.best_estimator_.predict(train)
        self.mse = mean_squared_error(label, predictions)
        self.rmse = np.sqrt(self.mse)
        self.rmsle = np.log(self.rmse)

    def display_best_estimator_eval_errors(self):
        print("MSE:", self.mse)
        print("RMSE:", self.rmse)
        print("RMSLE:", self.rmsle)

    def save_best_estimator(self, name):
        joblib.dump(self.search_cv.best_estimator_, name)

    def display_rmse_scores(self):
        print("Scores:", self.rmse_scores)
        print("Mean:", self.rmse_scores.mean())
        print("Standard Deviation:", self.rmse_scores.std())
