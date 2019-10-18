from sklearn.preprocessing import Binarizer, MaxAbsScaler, MinMaxScaler, Normalizer, RobustScaler, StandardScaler,\
    PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.base import TransformerMixin
import numpy as np


class DataProcessorType(TransformerMixin):
    # Just needed so deap can distinguish between data and feature processors, since SelectorMixin is
    # a subclass of TransformerMixin
    pass


class DummyTransformer(DataProcessorType):
    """
        This is a feature "selector" which does nothing,
        i.e. returns the original X values. The reason for this
        is so all pipelines can be treated as if they had a FS
        step.
    """
    def fit(self, X, y=None):
        # Doesnt need to do anything
        return self

    def transform(self, X):
        # Return original X values
        return X

    def __repr__(self):
        return "DummyTransformer()"

    __str__ = __repr__

processors = {
    DummyTransformer: {
    },

    Binarizer: {
        'threshold': [0., 0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 ,
       0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95, 1.]
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

    PolynomialFeatures: {

    },
}
