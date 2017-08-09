import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


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

    def grid_search(self, param_grid,
                    scoring='neg_mean_squared_error', **kwargs):
        m = self.model(**self.default_model_params)
        self.grid_search_cv = GridSearchCV(
            m, param_grid, scoring=scoring, **kwargs)
        self.grid_search_cv.fit(self.train, self.label)

    def display_rmse_scores(self):
        print("Scores:", self.rmse_scores)
        print("Mean:", self.rmse_scores.mean())
        print("Standard Deviation:", self.rmse_scores.std())
