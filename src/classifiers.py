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
from src.required import *
from sklearn.base import ClassifierMixin as Classifier
from src.combiners import Voting3Classifier, Voting5Classifier
import numpy as np
import random
from sklearn.svm import LinearSVC


def _create_estimator(method, *params):

    param_dict = {}

    for param in params:
        param_dict[param.name] = param.val

    # Required for linear SVC
    if method == LinearSVC:
        param_dict["dual"] = False

    model = method(**param_dict)

    return model


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


def add_estimators(pset, num_instances):
    """
    Adds the various classification algorithms that can be used as
    the final stage of pipeline

    :param pset:
    :return:
    """

    # Naive Bayes
    _add_estimator(pset, GaussianNB, [])

    # Decision Trees
    _add_estimator(pset, DecisionTreeClassifier, [])

    # Logistic Regression and Linear SVC
    _add_hyperparameter(pset, "C", types.CType, lambda: random.choice(np.logspace(-3, 2, 6)))
    _add_hyperparameter(pset, "Penalty", types.PenaltyType, lambda: random.choice(["l1", "l2"]))
    _add_estimator(pset, LogisticRegression, [types.CType, types.PenaltyType])
    _add_estimator(pset, LinearSVC, [types.CType, types.PenaltyType])


    # K-nn
    max_neighbors = min(50, num_instances - 1) # Upto a max of 50 neighbors, depending on train size
    _add_hyperparameter(pset, "K", types.KType, lambda: random.randrange(1, max_neighbors))
    _add_estimator(pset, KNeighborsClassifier, [types.KType])

    # Random Forests
    _add_hyperparameter(pset, "N", types.NumEstimatorsType, lambda: random.randrange(10, 150))
    _add_estimator(pset, RandomForestClassifier, [types.NumEstimatorsType])

    # Adaboost
    _add_estimator(pset, AdaBoostClassifier, [types.NumEstimatorsType])

    # XGBoost
    _add_hyperparameter(pset, "Booster", types.BoosterType, lambda: random.choice(["gbtree", "gblinear", "dart"]))
    _add_hyperparameter(pset, "Depth", types.DepthType, lambda: random.randint(2, 8))
    _add_hyperparameter(pset, "LR", types.LRType, lambda: random.choice([0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.5]))

    _add_estimator(pset, XGBClassifier, [types.BoosterType, types.DepthType, types.NumEstimatorsType, types.LRType])


def _add_hyperparameter(pset, name, ret_type, ephemeral_constant):
    pset.addEphemeralConstant(name, lambda: ret_type(ephemeral_constant()), ret_type)

def _add_estimator(pset, classifier, param_inputs):

    # Custom parameters
    if param_inputs:
        pset.addPrimitive(lambda *params: _create_estimator(classifier, *params), param_inputs,
                          Classifier, name=classifier.__name__ + "Terminal")

    # Default parameters
    pset.addTerminal(_create_estimator(classifier), Classifier, name=classifier.__name__ + "TerminalDefault")
