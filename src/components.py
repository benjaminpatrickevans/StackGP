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

import src.classifiers as classifiers
import src.customtypes as types
from src.required import *
from sklearn.base import ClassifierMixin as Classifier
from src.combiners import Voting3, Voting5
import numpy as np
import random

def add_combiners(pset):
    """
        Combiners turn pipeline(s) into
        a classifier. We can either have multiple
        pipelines combined with a voter, or just
        a single pipeline.

    :param pset:
    :return:
    """

    # Single pipeline. Takes a list of pipeline steps, and returns a pipeline (treated as a classifier)
    pset.addPrimitive(lambda prev_steps: Pipeline(steps=prev_steps), [types.PipelineStump], Classifier,
                      name="MakePipeline")

    # Otherwise we want to be able to combine pipelines/classifiers into a single voting classifier
    pset.addPrimitive(lambda p1, p2, p3: Voting3(p1, p2, p3), [Classifier] * 3, Classifier,
                      name="Voting3")

    pset.addPrimitive(lambda p1, p2, p3, p4, p5: Voting5(p1, p2, p3, p4, p5), [Classifier] * 5, Classifier,
                      name="Voting5")

def add_classifiers(pset, num_instances, num_features, num_classes):
    """
    Adds the various classification algorithms that can be used as
    the final stage of pipeline

    :param pset:
    :return:
    """

    # Naive Bayes
    _add_classifier(pset, GaussianNB, [])

    # Decision Trees
    _add_classifier(pset, DecisionTreeClassifier, [])

    # Logistic Regression and Linear SVC
    _add_terminal(pset, "C", types.CType, lambda: random.choice(np.logspace(-3, 2, 6)))
    _add_terminal(pset, "Penalty", types.PenaltyType, lambda: random.choice(["l1", "l2"]))
    _add_classifier(pset, LogisticRegression, [types.CType, types.PenaltyType])
    _add_classifier(pset, LinearSVC, [types.CType, types.PenaltyType])

    # SGD Classifier
    '''
    _add_terminal(pset, "Loss", types.LossType, lambda: random.choice(["hinge", "log", "modified_huber", "perceptron"]))
    _add_terminal(pset, "Alpha", types.AlphaType, lambda: random.choice([10 ** x for x in range(-6, 1)]))
    _add_terminal(pset, "Iter", types.IterType, lambda: random.choice([5, 10, 100, 1000]))
    _add_classifier(pset, SGDClassifier, [types.PenaltyType, types.LossType, types.AlphaType, types.IterType],
                    input_types)
    '''

    # K-nn
    max_neighbors = min(50, num_instances - 1) # Upto a max of 50 neighbors, depending on train size
    _add_terminal(pset, "K", types.KType, lambda: random.randrange(1, max_neighbors))
    _add_classifier(pset, KNeighborsClassifier, [types.KType])

    # Random Forests
    _add_terminal(pset, "N", types.NumEstimatorsType, lambda: random.randrange(10, 150))
    _add_classifier(pset, RandomForestClassifier, [types.NumEstimatorsType])

    # Adaboost
    _add_classifier(pset, AdaBoostClassifier, [types.NumEstimatorsType])

    # XGBoost
    _add_terminal(pset, "Booster", types.BoosterType, lambda: random.choice(["gbtree", "gblinear", "dart"]))
    _add_terminal(pset, "Depth", types.DepthType, lambda: random.randint(2, 8))
    _add_terminal(pset, "LR", types.LRType, lambda: random.choice([0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.5]))

    _add_classifier(pset, XGBClassifier, [types.BoosterType, types.DepthType, types.NumEstimatorsType, types.LRType])


def _add_terminal(pset, name, ret_type, ephemeral_constant):
    pset.addEphemeralConstant(name, lambda: ret_type(ephemeral_constant()), ret_type)


def _add_classifier(pset, classifier, param_inputs):

    # Custom parameters
    pset.addPrimitive(lambda *params: classifiers.base([], classifier, *params), param_inputs,
                      types.PipelineStump, name=classifier.__name__ + "Terminal")

    # Default parameters
    pset.addPrimitive(lambda: classifiers.base([], classifier), [], types.PipelineStump,
                      name=classifier.__name__ + "TerminalDefault")