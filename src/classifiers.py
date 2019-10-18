from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.base import ClassifierMixin as ClassifierType
from sklearn.pipeline import Pipeline
from src.combiners import VotingBaseClassifier, StackingBaseClassifier
import random
import src.customtypes as types

classifier_map = {
    # Classifiers
    GaussianNB: {
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
        'max_features': [0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 , 0.55,
       0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95, 1.  ],
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },

    RandomForestClassifier: {
        'n_estimators': [100],
        'criterion': ["gini", "entropy"],
        'max_features': [0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 , 0.55,
       0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95, 1.],
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
        'subsample': [0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 , 0.55,
       0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95, 1.  ],
        'max_features': [0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 , 0.55,
       0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95, 1.  ],
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
        'subsample': [0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 , 0.55,
       0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95, 1.  ],
        'min_child_weight': range(1, 21),
        'nthread': [1]
    },
}

def add_combiners(pset):
    """
        Voters turn individual classifiers into a
        majority-vote ensemble

    :param pset:
    :return:
    """

    # Combine classifiers into a single voting classifier. Takes in 3 or 5 classifiers and returns the majority vote
    pset.addPrimitive(lambda p1, p2, p3: VotingBaseClassifier([p1, p2, p3]), [ClassifierType] * 3, ClassifierType,
                      name="Voting3")

    pset.addPrimitive(lambda p1, p2, p3, p4, p5: VotingBaseClassifier([p1, p2, p3, p4, p5]), [ClassifierType] * 5, ClassifierType,
                      name="Voting5")

    # For recreation
    pset.context["VotingClassifier"] = VotingClassifier

    # This is to specify  whether or not to use the original features alongside the predictions in the meta classifier
    # It basically just acts as a wrapper around boolean, but since we cant make a subclass of bool we must do some
    # work to "wrap" the true/false values into a new specialised type.
    stackfeature_type = type("StackingFeatureType", (), {'name': "StackingFeatureType", '__init__': types.param_init,
                                                          '__str__': types.param_str, '__repr__': types.param_str,
                                                         'range': [True, False]})
    pset.context["StackingFeatureType"] = stackfeature_type

    # Whether or not to use the original features alongside the predictions in the meta classifier
    pset.addEphemeralConstant("StackingFeatures", lambda: stackfeature_type(random.choice([True, False])),
                              stackfeature_type)

    # Uses logistic regression to perform stacking
    pset.addPrimitive(lambda p1, p2, p3, use_features:
                      StackingBaseClassifier([p1, p2, p3], use_features=use_features),
                      [ClassifierType] * 3 + [stackfeature_type], ClassifierType, name="LRStacking3")

    pset.addPrimitive(lambda p1, p2, p3, p4, p5, use_features:
                      StackingBaseClassifier([p1, p2, p3, p4, p5], use_features=use_features),
                      [ClassifierType] * 5 + [stackfeature_type], ClassifierType, name="LRStacking5")

    # For recreation
    pset.context["StackingCVClassifier"] = StackingCVClassifier
