"""
    This file contains the function and terminal
    set used for the boosted pipelines, including
    all feature selection methods, classifiers etc.

    In the future it would be ideal to convert this
    to an arg file to allow for easier expansion
    for end users.

    NOTE: Your IDE may report missing classes, because
    many of these are created dynamically to allow for
    easy expansion in the future (see customtypes.py)
"""
import src.customtypes as types
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.base import ClassifierMixin as Classifier
from src.combiners import Voting3Classifier, Voting5Classifier
import numpy as np

classifier_map = {
        # Naive Bayes
        GaussianNB: {},

        # Decision Trees
        DecisionTreeClassifier: {},

        # Logistic Regression and Linear SVC
        LogisticRegression:  {
            "C": np.logspace(-3, 2, 6),
            "penalty": ["l1", "l2"]
        },

        LinearSVC:  {
            "C": np.logspace(-3, 2, 6),
            "penalty": ["l1", "l2"],
            "dual": [False]
        },

        KNeighborsClassifier: {
            "n_neighbors": range(1, 50)
        },

        RandomForestClassifier: {
            "n_estimators": range(10, 150)
        },

        AdaBoostClassifier: {
            "n_estimators": range(10, 150)
        },

        XGBClassifier: {
            "n_estimators": range(10, 150),
            "booster": ["gbtree", "gblinear", "dart"],
            "max_depth": range(2, 8),
            "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.5]
            }
}

def add_voters(pset):
    """
        Voters turn individual classifiers into a
        majority-vote ensemble

    :param pset:
    :return:
    """

    # Combine classifiers into a single voting classifier
    pset.addPrimitive(lambda p1, p2, p3: Voting3Classifier(p1, p2, p3), [Classifier] * 3, Classifier,
                      name="Voting3")

    pset.addPrimitive(lambda p1, p2, p3, p4, p5: Voting5Classifier(p1, p2, p3, p4, p5), [Classifier] * 5, Classifier,
                      name="Voting5")

    pset.context["VotingClassifier"] = VotingClassifier
