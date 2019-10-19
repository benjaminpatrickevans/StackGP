from src.base import Base
from src import classifiers, components, feature_processors, data_processors
from sklearn.base import ClassifierMixin, TransformerMixin

class StackGPClassifier(Base):

    def __init__(self, pop_size=100, max_run_time_mins=60, max_eval_time_mins=5, max_depth=17, n_jobs=1,
                 verbose=0, random_state=0):
        self.est_type = ClassifierMixin
        self.scoring_fn = "f1_weighted"
        super().__init__(pop_size=pop_size, max_run_time_mins=max_run_time_mins, max_eval_time_mins=max_eval_time_mins,
                         max_depth=max_depth, n_jobs=n_jobs, verbose=verbose, random_state=random_state)

    def _add_components(self, pset, num_features):

        # Data preprocessors
        data_preprocessors = data_processors.processors
        if num_features > 32:
            # Polynomial features will create >1000 features in this case, so we dont want that as we could
            # risk overfitting.
            data_preprocessors.pop(data_processors.PolynomialFeatures)

        components.add_components(pset, data_preprocessors, data_processors.DataProcessorType,
                                  prev_step_type=None)

        # Feature selectors
        components.add_components(pset, feature_processors.processors, feature_processors.FeatureProcessorType,
                                  prev_step_type=data_processors.DataProcessorType)

        # Classifiers
        components.add_components(pset, classifiers.classifier_map, classifiers.ClassifierType,
                                  prev_step_type=feature_processors.FeatureProcessorType)

        # Combiners
        classifiers.add_combiners(pset)




