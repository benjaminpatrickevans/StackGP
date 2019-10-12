from src.base import Base
from src import classifiers, components, feature_processors, data_processors
from sklearn.base import ClassifierMixin, TransformerMixin
from sklearn.feature_selection.base import SelectorMixin

class StackGPClassifier(Base):

    def __init__(self, pop_size=100, max_run_time_mins=60, crs_rate=0.8, mut_rate=0.2, max_depth=17, n_jobs=1,
                 verbose=0, random_state=0):
        self.est_type = ClassifierMixin
        self.scorer = "f1_weighted"
        super().__init__(pop_size=pop_size, max_run_time_mins=max_run_time_mins, crs_rate=crs_rate, mut_rate=mut_rate,
                         max_depth=max_depth, n_jobs=n_jobs, verbose=verbose, random_state=random_state)

    def _add_components(self, pset):

        # Data preprocessors
        components.add_components(pset, data_processors.processors, TransformerMixin, prev_step_type=None)

        # Feature selectors
        components.add_components(pset, feature_processors.processors, SelectorMixin, prev_step_type=TransformerMixin)

        # Classifiers
        components.add_components(pset, classifiers.classifier_map, ClassifierMixin, prev_step_type=SelectorMixin)

        # Combiners
        classifiers.add_combiners(pset)




