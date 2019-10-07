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
from src.required_classifiers import *
from sklearn.base import ClassifierMixin as Classifier
from src.combiners import Voting3Classifier, Voting5Classifier
import numpy as np
import random
from sklearn.svm import LinearSVC

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

    # For sharing parameter types
    all_parameter_types = {}

    for classifier in classifier_map:
        classifier_params = classifier_map[classifier]

        inputs = []
        for param in classifier_params:
            param_name = str(param)

            if param_name in all_parameter_types:
                # Already made this type, reuse it. This is so we can cross types over
                param_type = all_parameter_types[param_name]
            else:
                # Must add the type
                value_range = classifier_params[param]

                param_type = type(param, (), {'name': param, '__init__': types.param_init, '__str__': types.param_str,
                                    '__repr__': types.param_str})

                _add_hyperparameter(pset, param_name, param_type, value_range)

                # For recreating the types
                pset.context[param] = param_type

                all_parameter_types[param_name] = param_type

            inputs.append(param_type)

        _add_estimator(pset, classifier, inputs)

def _add_hyperparameter(pset, name, ret_type, value_range):
    pset.addEphemeralConstant(name, lambda: ret_type(random.choice(value_range)), ret_type)


def _add_estimator(pset, classifier, param_inputs):

    # Custom parameters
    if param_inputs:
        pset.addPrimitive(lambda *params: _create_estimator(classifier, *params), param_inputs,
                          Classifier, name=classifier.__name__ + "Terminal")

    # Default parameters
    pset.addTerminal(_create_estimator(classifier), Classifier, name=classifier.__name__ + "TerminalDefault")
