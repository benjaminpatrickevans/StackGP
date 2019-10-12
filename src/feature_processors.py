from sklearn.feature_selection import SelectPercentile, SelectFromModel
from sklearn.base import TransformerMixin
from sklearn.feature_selection import mutual_info_classif, chi2, f_classif
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection.base import SelectorMixin
import random

class FeatureProcessorType(SelectorMixin):
    pass

class DummySelector(FeatureProcessorType):
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


class RandomSelector(FeatureProcessorType):
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
        # Ensure we select atleast 1 feature
        self.num_features_to_select = max(1, int(self.num_features * (self.percentile / 100)))

        return self

    def __repr__(self):
        return "RandomSelector(percentile=" + repr(self.percentile) + ", "+ "seed=" + repr(self.seed) + ")"

    __str__ = __repr__


class SelectPercentileBase(SelectPercentile):
    """
    SelectPercnetile with a recreatable repr
    and ensuring we select atleast one feature.
    """

    def __init__(self, scorer, percentile):
        super().__init__(scorer, percentile=percentile)

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.percentile) + ")"

    __str__ = __repr__

    def fit(self, X, y):
        # Just does a safety check to ensure we select atleast 1 feature
        num_features = X.shape[1]

        num_features_to_select = int(num_features * (self.percentile / 100))

        if num_features_to_select == 0:
            # The minimum percentage to give 1 feature
            self.percentile = int(1 / num_features * 100)

        return super().fit(X, y)


class SelectFClassif(SelectPercentileBase):
    def __init__(self, percentile):
        super().__init__(f_classif, percentile=percentile)


class SelectMutualInfo(SelectPercentileBase):
    def __init__(self, percentile):
        super().__init__(mutual_info_classif, percentile=percentile)


class SelectChi2(SelectPercentileBase):
    def __init__(self, percentile):
        super().__init__(chi2, percentile=percentile)


class SelectPercentileFromModel(SelectFromModel):

    def __init__(self, estimator, percentile):
        self.percentile = percentile
        super().__init__(estimator=estimator, threshold=-np.inf)

    def fit(self, X, y=None, **fit_params):
        num_features = X.shape[1]

        # How many features to select. Make sure we select atleast 1
        self.max_features = max(1, int(num_features * (self.percentile / 100)))

        return super().fit(X, y, **fit_params)

# Selectors
processors = {
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

    SelectPercentileFromModel: {
        'estimator': [DecisionTreeClassifier(), RandomForestClassifier(), ExtraTreesClassifier(),
                      LinearSVC(penalty="l1", dual=False), LogisticRegression(penalty="l1")],
        'percentile': range(5, 101, 5)
    }

}
