from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.base import ClassifierMixin as Classifier
from src.combiners import Voting3Classifier, Voting5Classifier, Stacking3Classifier, Stacking5Classifier
import numpy as np
import random
import src.customtypes as types

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

    # Combine classifiers into a single voting classifier. Takes in 3 or 5 classifiers and returns the majority vote
    pset.addPrimitive(lambda p1, p2, p3: Voting3Classifier(p1, p2, p3), [Classifier] * 3, Classifier, name="Voting3")

    pset.addPrimitive(lambda p1, p2, p3, p4, p5: Voting5Classifier(p1, p2, p3, p4, p5), [Classifier] * 5, Classifier,
                      name="Voting5")

    # For recreation
    pset.context["VotingClassifier"] = VotingClassifier

    # This is to specify  whether or not to use the original features alongside the predictions in the meta classifier
    # It basically just acts as a wrapper around boolean, but since we cant make a subclass of bool we must do some
    # work to "wrap" the true/false values into a new specialised type.
    stackfeature_type = type("StackingFeatureType", (), {'name': "StackingFeatureType", '__init__': types.param_init,
                                                          '__str__': types.param_str, '__repr__': types.param_str})
    pset.context["StackingFeatureType"] = stackfeature_type

    # Whether or not to use the original features alongside the predictions in the meta classifier
    pset.addEphemeralConstant("StackingFeatures", lambda: stackfeature_type(random.choice([True, False])),
                              stackfeature_type)

    # Uses a meta level classifier to perform stacking, i.e., train the meta classifier on the predictions of the inputs
    pset.addPrimitive(lambda meta, p1, p2, p3, use_features:
                      Stacking3Classifier(p1, p2, p3, meta_classifier=meta, use_features=use_features),
                      [Classifier] * 4 + [stackfeature_type], Classifier, name="Stacking3")

    pset.addPrimitive(lambda meta, p1, p2, p3, p4, p5, use_features:
                      Stacking5Classifier(p1, p2, p3, p4, p5, meta_classifier=meta, use_features=use_features),
                      [Classifier] * 6 + [stackfeature_type], Classifier, name="Stacking5")

    # For recreation
    pset.context["StackingCVClassifier"] = StackingCVClassifier
