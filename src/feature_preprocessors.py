from sklearn.feature_selection import SelectPercentile, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, chi2, f_classif

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


def ranking_selector(ranker, percentile, prev_steps=None):

    if prev_steps is None:
        prev_steps = []

    # Select the top percentile based on given ranker and percentile
    selector = ranker(percentile=percentile.val)
    return prev_steps + [("fp", selector)]


def tree_selector(model, multiplier, prev_steps=None):

    if prev_steps is None:
        prev_steps = []

    # Select only the features above the given threshold
    selector = SelectFromModel(model, threshold=str(multiplier.val) + "*mean")

    return prev_steps + [("fp", selector)]


def l1_selector(model, C, prev_steps=None):

    if prev_steps is None:
        prev_steps = []

    # Need to set the C parameter before performing selection
    model.C = C.val
    selector = SelectFromModel(model)

    return prev_steps + [("fp", selector)]


def PCA_reduction(percentile, num_features, prev_steps=None):

    if prev_steps is None:
        prev_steps = []

    num_components = int(percentile.val / 100 * num_features)
    selector = PCA(n_components=num_components)

    return prev_steps + [("fp", selector)]