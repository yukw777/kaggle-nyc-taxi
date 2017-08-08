import numpy as np
from sklearn.model_selection import cross_val_score


class ModelEvaluator():

    def __init__(self, model, train, label, default_params_for_cv={}):
        self.model = model
        self.train = train
        self.label = label
        self.default_params_for_cv = default_params_for_cv

    def cross_validate(self, scoring="neg_mean_squared_error", **kwargs):
        m = self.model(**self.default_params_for_cv)
        self.scores = cross_val_score(
            m, self.train, y=self.label, scoring=scoring, **kwargs)
        self.rmse_scores = np.sqrt(-self.scores)

    def display_rmse_scores(self):
        print("Scores:", self.rmse_scores)
        print("Mean:", self.rmse_scores.mean())
        print("Standard Deviation:", self.rmse_scores.std())
