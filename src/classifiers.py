from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
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
    # Classifiers
    GaussianNB: {
    },

    BernoulliNB: {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    },

    MultinomialNB: {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    },

    DecisionTreeClassifier: {
        'criterion': ["gini", "entropy"],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21)
    },

    ExtraTreesClassifier: {
        'n_estimators': [100],
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },

    RandomForestClassifier: {
        'n_estimators': [100],
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf':  range(1, 21),
        'bootstrap': [True, False]
    },

    GradientBoostingClassifier: {
        'n_estimators': [100],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'max_features': np.arange(0.05, 1.01, 0.05)
    },

    KNeighborsClassifier: {
        'n_neighbors': range(1, 101),
        'weights': ["uniform", "distance"],
        'p': [1, 2]
    },

    LinearSVC: {
        'penalty': ["l1", "l2"],
        'dual': [False],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.]
    },

    LogisticRegression: {
        'penalty': ["l1", "l2"],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
    },

    XGBClassifier: {
        'n_estimators': [100],
        'max_depth': range(1, 11),
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'subsample': np.arange(0.05, 1.01, 0.05),
        'min_child_weight': range(1, 21),
        'nthread': [1]
    },
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
