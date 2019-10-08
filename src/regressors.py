from sklearn.linear_model import ElasticNetCV, LassoLarsCV, RidgeCV, LinearRegression
from sklearn.ensemble import VotingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor, \
    RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor
from sklearn.base import RegressorMixin as Regressor
from src.combiners import Voting3Regressor, Voting5Regressor
import numpy as np

estimators = {

    #ElasticNetCV : {
    #    'l1_ratio': np.arange(0.0, 1.01, 0.05),
    #    'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    #},

    ExtraTreesRegressor: {
        'n_estimators': [100],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },

    GradientBoostingRegressor: {
        'n_estimators': [100],
        'loss': ["ls", "lad", "huber", "quantile"],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'max_features': np.arange(0.05, 1.01, 0.05),
        'alpha': [0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    },

    AdaBoostRegressor: {
        'n_estimators': [100],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'loss': ["linear", "square", "exponential"]
    },

    DecisionTreeRegressor: {
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21)
    },

    KNeighborsRegressor: {
        'n_neighbors': range(1, 101),
        'weights': ["uniform", "distance"],
        'p': [1, 2]
    },

    LassoLarsCV: {
        'normalize': [True, False]
    },


    RandomForestRegressor: {
        'n_estimators': [100],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },

    RidgeCV: {
    },

    LinearRegression: {
    },

    XGBRegressor: {
        'n_estimators': [100],
        'max_depth': range(1, 11),
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'subsample': np.arange(0.05, 1.01, 0.05),
        'min_child_weight': range(1, 21),
        'nthread': [1],
        'objective': ['reg:squarederror']
    }
}

def add_voters(pset):
    """
        Voters perform the mean vote of their inputs.
        Possible voters are VotingRegressors with
        either 3 or 5 inputs.
    :param pset:
    :return:
    """
    # Combine classifiers into a single voting classifier
    pset.addPrimitive(lambda p1, p2, p3: Voting3Regressor(p1, p2, p3), [Regressor] * 3, Regressor,
                      name="Voting3")

    pset.addPrimitive(lambda p1, p2, p3, p4, p5: Voting5Regressor(p1, p2, p3, p4, p5), [Regressor] * 5, Regressor,
                      name="Voting5")

    pset.context["VotingRegressor"] = VotingRegressor