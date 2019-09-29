"""
    This class provides some helper functionality that would be ideal
    but is not currently included in sklearn.

"""
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from mlxtend.classifier import StackingCVClassifier
from sklearn.preprocessing import LabelEncoder

class DataFrameImputer(TransformerMixin):
    '''
        Source: https://stackoverflow.com/questions/25239958/impute-categorical-missing-values-in-scikit-learn
    '''

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with mean of column.

        """

    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


class StackingCV(StackingCVClassifier):

    def fit(self, X, y):
        self.le = LabelEncoder()
        y = self.le.fit_transform(y)

        super().fit(X, y)

    def predict(self, X):
        predictions = super().predict(X)
        return self.le.inverse_transform(predictions)

