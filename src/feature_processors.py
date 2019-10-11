from sklearn.feature_selection import SelectPercentile, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, chi2, f_classif
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection.base import SelectorMixin
import random


class DummySelector(SelectorMixin):
    """
        This is a feature "selector" which does nothing,
        i.e. returns the original X values. The reason for this
        is so all pipelines can be treated as if they had a FS
        step.
    """

    def _get_support_mask(self):
        # All values selected
        mask = np.ones(self.num_features, dtype=bool)
        return mask

    def fit(self, X, y):
        self.num_features = X.shape[1]
        return self

    def __repr__(self):
        return "DummySelector()"

    __str__ = __repr__


class RandomSelector(SelectorMixin):
    """
        This is a feature selector which returns a random subset
        of the original features.
    """

    def __init__(self, percentile, seed):
        self.percentile = percentile
        self.seed = seed

    def _get_support_mask(self):
        # To begin with no features selected
        mask = np.zeros(self.num_features, dtype=bool)

        # Reproducability of the randomness
        random.seed(self.seed)

        # Select random feature indices
        random_features = random.sample(range(0, self.num_features), self.num_features_to_select)

        # Flip these on in our mask
        mask[random_features] = True

        return mask

    def fit(self, X, y):
        self.num_features = X.shape[1]
        self.num_features_to_select = int(self.num_features * (self.percentile / 100))
        return self

    def __repr__(self):
        return "RandomSelector(percentile=" + repr(self.percentile) + ", "+ "seed=" + repr(self.seed) + ")"

    __str__ = __repr__


# Selectors
processors = {
    PCA: {
        # TODO: Custom PCA with percentage
    },

    #SelectPercentile: {
    #    'percentile': range(5, 91, 5),
    #    'score_func': [f_classif, mutual_info_classif, chi2]
    #},

    #SelectFromModel: {
    #    'threshold': np.arange(0, 1.01, 0.05),
    #    'estimator': [DecisionTreeClassifier(), RandomForestClassifier(), ExtraTreesClassifier(),
    #                  LinearSVC(penalty="l1", dual=False), LogisticRegression(penalty="l1")]
    #},

    # Dummy for skipping feature preprocessing
    DummySelector: {},

    # Random Features for introducing diversity into the ensembles
    RandomSelector: {
        'seed': range(100),
        'percentile': range(5, 51, 5)
    }

}



