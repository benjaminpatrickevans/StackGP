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

import src.feature_preprocessors as fp
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

    '''
    meta_learners = [LogisticRegression]
    num_base_learners = [3, 5]

    for learner in meta_learners:
        for num_base in num_base_learners:
            pset.addPrimitive(lambda *clf, meta=learner: StackingCV(classifiers=[*clf],
                                                                              meta_classifier=meta()),
                              [Classifier] * num_base, Classifier, name="LRStackCV"+str(num_base))
    '''


def add_classifiers(pset, num_instances, num_features, num_classes):
    """
    Adds the various classification algorithms that can be used as
    the final stage of pipeline

    :param pset:
    :return:
    """

    # Should allow inputs from any of the previous steps to have flexible pipelines
    input_types = [types.FeatureProcessOutput, types.DataProcessOutput]

    # Naive Bayes
    _add_classifier(pset, GaussianNB, [], input_types)

    # Decision Trees
    _add_classifier(pset, DecisionTreeClassifier, [], input_types)

    # Logistic Regression and Linear SVC
    _add_terminal(pset, "C", types.CType, lambda: random.choice(np.logspace(-3, 2, 6)))
    _add_terminal(pset, "Penalty", types.PenaltyType, lambda: random.choice(["l1", "l2"]))
    _add_classifier(pset, LogisticRegression, [types.CType, types.PenaltyType], input_types)
    _add_classifier(pset, LinearSVC, [types.CType, types.PenaltyType], input_types)

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
    _add_classifier(pset, KNeighborsClassifier, [types.KType], input_types)

    # Random Forests
    _add_terminal(pset, "N", types.NumEstimatorsType, lambda: random.randrange(10, 150))
    _add_classifier(pset, RandomForestClassifier, [types.NumEstimatorsType], input_types)

    # Adaboost
    _add_classifier(pset, AdaBoostClassifier, [types.NumEstimatorsType], input_types)

    # XGBoost
    _add_terminal(pset, "Booster", types.BoosterType, lambda: random.choice(["gbtree", "gblinear", "dart"]))
    _add_terminal(pset, "Depth", types.DepthType, lambda: random.randint(2, 8))
    _add_terminal(pset, "LR", types.LRType, lambda: random.choice([0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.5]))

    _add_classifier(pset, XGBClassifier, [types.BoosterType, types.DepthType, types.NumEstimatorsType, types.LRType],
                    input_types)


def add_feature_preprocessors(pset, all_non_negative_values, num_features):
    """
        Adds the various feature preprocessors
    :return:
    """


    scorers = [fp.SelectFClassif, fp.SelectMutualInfo] # fp.SelectFisher]

    if all_non_negative_values:
        # Only relevant for positive values
        scorers.append(fp.SelectChi2)

    trees = [DecisionTreeClassifier(), RandomForestClassifier(), ExtraTreesClassifier()]
    l1_estimators = [LinearSVC(penalty="l1", dual=False), LogisticRegression(penalty="l1")]

    # Select 1% of features
    min_percentage = 1

    # We want to round up if 1% of the features is <1 feature. This is to ensure we never select 0 features
    if num_features / 100 < 1:
        # The minimum percentage that gives us atleast 1 feature
        min_percentage = int(1 / num_features * 100)

    # Just a wrapper around int, needed for STGP. Specifies the percentage of features to keep
    _add_terminal(pset, "Percentile", types.PercentileType, lambda: random.randrange(min_percentage, 101, 5))

    # Controls the C value for LinearSVC and LogisticRegression
    _add_terminal(pset, "SelectC", types.CType_Selection, lambda: random.choice(np.logspace(-3, 2, 6)))

    # For selecting features from a model. Used for setting the threshold, i.e. x * mean
    _add_terminal(pset, "Threshold", types.ThresholdType, lambda: random.choice([0.5, 0.75, 1.0, 1.25, 2, 2.5]))

    # Add the various selection functions

    input_types = [[types.DataProcessOutput], []]

    for idx, input_type in enumerate(input_types):

        # Filter based
        for selector in scorers:
            pset.addPrimitive(lambda percentile, prev_steps=None, selection_method=selector:
                              fp.ranking_selector(selection_method, percentile, prev_steps),
                              [types.PercentileType] + input_type, types.FeatureProcessOutput,
                              name="SelectScorer"+selector.__name__ + str(idx))

        # Tree-based
        for selector in trees:
            pset.addPrimitive(lambda multiplier, prev_steps=None, selection_method=selector:
                              fp.tree_selector(selection_method, multiplier, prev_steps),
                              [types.ThresholdType] + input_type, types.FeatureProcessOutput,
                              name="SelectTree" + selector.__class__.__name__ + str(idx))

        # Penalty based
        for selector in l1_estimators:
            pset.addPrimitive(lambda C, prev_steps=None, selection_method=selector:
                              fp.l1_selector(selection_method, C, prev_steps),
                              [types.CType_Selection] + input_type, types.FeatureProcessOutput,
                              name="SelectSparse"+selector.__class__.__name__ + str(idx))

        # PCA
        pset.addPrimitive(lambda percentile, prev_steps=None: fp.PCA_reduction(percentile, num_features, prev_steps),
                          [types.PercentileType] + input_type, types.FeatureProcessOutput,
                          name="SelectPCA" + str(idx))



def add_data_preprocessors(pset):
    """
        Adds scaling and imputing of missing values into
        the pipeline
    :param pset:
    :return:
    """

    scalers = [StandardScaler, RobustScaler, MinMaxScaler]

    def add_scaler(prev_steps, scaler):
        if prev_steps is None:
            prev_steps = []

        return prev_steps + [("dp", scaler())]

    for method in scalers:
        pset.addPrimitive(lambda prev_steps=None, scaler=method: add_scaler(prev_steps, scaler), [],
                          types.DataProcessOutput, name=method.__name__)


def _add_terminal(pset, name, ret_type, ephemeral_constant):
    pset.addEphemeralConstant(name, lambda: ret_type(ephemeral_constant()), ret_type)


def _add_classifier(pset, classifier, param_inputs, custom_inputs):

    for idx, input_type in enumerate(custom_inputs):

        inputs = [input_type] + param_inputs

        # Add as a primitive, where we have custom params
        pset.addPrimitive(lambda prev_steps, *params: classifiers.base(prev_steps, classifier, *params), inputs,
                          types.PipelineStump, name=classifier.__name__+str(idx))

        # Add as a primitive where we have no params
        pset.addPrimitive(lambda prev_steps: classifiers.base(prev_steps, classifier), [input_type],
                          types.PipelineStump, name=classifier.__name__ + str(idx) + "Default")

    # Then we need to add the classifiers with no previous steps. Cant be done above because would require
    # kwargs and splat operator, which would not work.

    # Case when there are no previous steps, but we have params
    pset.addPrimitive(lambda *params: classifiers.base([], classifier, *params), param_inputs,
                      types.PipelineStump, name=classifier.__name__ + "Terminal")

    # Case when there are no previous steps, and default params
    pset.addPrimitive(lambda: classifiers.base([], classifier), [], types.PipelineStump,
                      name=classifier.__name__ + "TerminalDefault")