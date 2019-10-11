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
from src import customtypes as types


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


class SelectPercentileBase(SelectPercentile):
    """
    SelectPercnetile with a recreatable repr.
    """

    def __init__(self, scorer, percentile):
        super().__init__(scorer, percentile=percentile)

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.percentile) + ")"

    __str__ = __repr__


class SelectFClassif(SelectPercentileBase):
    def __init__(self, percentile):
        super().__init__(f_classif, percentile=percentile)


class SelectMutualInfo(SelectPercentileBase):
    def __init__(self, percentile):
        super().__init__(mutual_info_classif, percentile=percentile)


class SelectChi2(SelectPercentileBase):
    def __init__(self, percentile):
        super().__init__(chi2, percentile=percentile)


# Selectors
processors = {
    PCA: {
        # TODO: Custom PCA with percentage
    },

    # Dummy for skipping feature preprocessing
    DummySelector: {},

    # Random Features for introducing diversity into the ensembles
    RandomSelector: {
        'seed': range(100),
        'percentile': range(5, 51, 5)
    },

    SelectFClassif: {
        'percentile': range(5, 101, 5)
    },

    SelectMutualInfo: {
        'percentile': range(5, 101, 5)
    },

    SelectChi2: {
        'percentile': range(5, 101, 5)
    },

    SelectFromModel: {
        'estimator': [DecisionTreeClassifier(), RandomForestClassifier(), ExtraTreesClassifier(),
                      LinearSVC(penalty="l1", dual=False), LogisticRegression(penalty="l1")],
        'threshold': np.arange(0, 1.01, 0.05)
    }

}
