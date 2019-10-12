from sklearn.preprocessing import Binarizer, MaxAbsScaler, MinMaxScaler, Normalizer, RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import TransformerMixin
import numpy as np

class DummyTransformer(TransformerMixin):
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
        return "DummyTransformer()"

    __str__ = __repr__

processors = {
    Binarizer: {
        'threshold': np.arange(0.0, 1.01, 0.05)
    },

    Normalizer: {
        'norm': ['l1', 'l2', 'max']
    },

    MaxAbsScaler: {
    },

    MinMaxScaler: {
    },

    PCA: {
    },

    RobustScaler: {
    },

    StandardScaler: {
    },
}