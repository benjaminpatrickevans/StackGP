from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn import base
import numpy as np
from sklearn.externals import six
from mlxtend.classifier import StackingCVClassifier

# Sklearn has hardcoded limits for repr outputs which is obviously not desirable for recreating, so this needs to be
# overriden without the limits

def _pprint(params, offset=0, printer=repr):
    """
    From: https://github.com/scikit-learn/scikit-learn/blob/51a765a/sklearn/base.py
    With line 142-143 removed
    """
    # Do a multi-line justified repr:
    options = np.get_printoptions()
    np.set_printoptions(precision=5, threshold=64, edgeitems=2)
    params_list = list()
    this_line_length = offset
    line_sep = ',\n' + (1 + offset // 2) * ' '
    for i, (k, v) in enumerate(sorted(six.iteritems(params))):
        if type(v) is float:
            # use str for representing floating point numbers
            # this way we get consistent representation across
            # architectures and versions.
            this_repr = '%s=%s' % (k, str(v))
        else:
            # use repr of the rest
            this_repr = '%s=%s' % (k, printer(v))
        if i > 0:
            if (this_line_length + len(this_repr) >= 75 or '\n' in this_repr):
                params_list.append(line_sep)
                this_line_length = len(line_sep)
            else:
                params_list.append(', ')
                this_line_length += 2
        params_list.append(this_repr)
        this_line_length += len(this_repr)

    np.set_printoptions(**options)
    lines = ''.join(params_list)
    # Strip trailing space to avoid nightmare in doctests
    lines = '\n'.join(l.rstrip(' ') for l in lines.split('\n'))
    return lines

# To nesure we dont crop the repr methods
base._pprint = _pprint


class StackingBaseClassifier(StackingCVClassifier):

    def __init__(self, classifiers, meta_classifier, use_features):
        super().__init__(classifiers=classifiers, meta_classifier=meta_classifier,
                         use_features_in_secondary=use_features)

    def __repr__(self):
        return "StackingCVClassifier(classifiers=" + repr(self.classifiers) + ", meta_classifier="\
               + repr(self.meta_classifier) + ", use_features_in_secondary=" + repr(self.use_features_in_secondary) + ")"

    __str__ = __repr__


class VotingBaseClassifier(VotingClassifier):

    def __init__(self, estimators):
        named_estimators = [("Est" + str(idx), clf) for idx, clf in enumerate(estimators)]
        # They must be named
        super().__init__(estimators=named_estimators)

    def __repr__(self):
        # When recreating, we can just make a VotingClassifier now
        return "VotingClassifier(estimators=" + repr(self.estimators) + ")"

    __str__ = __repr__


class VotingBaseRegressor(VotingRegressor):

    def __init__(self, estimators):
        named_estimators = [("Est" + str(idx), clf) for idx, clf in enumerate(estimators)]
        super().__init__(estimators=named_estimators)

    def __repr__(self):
        # When recreating, we can just make a VotingClassifier now
        return "VotingRegressor(estimators=" + repr(self.estimators) + ")"

    __str__ = __repr__