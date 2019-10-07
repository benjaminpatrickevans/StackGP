from src.base import Base
from src import classifiers
from src.required_classifiers import *  # Needed for eval to recreate individuals, do not delete

class StackGPClassifier(Base):

    def __init__(self, pop_size=100, max_run_time_mins=60, crs_rate=0.8, mut_rate=0.2, max_depth=17, n_jobs=1,
                 verbose=0, random_state=0):
        super().__init__(pop_size=pop_size, max_run_time_mins=max_run_time_mins, crs_rate=crs_rate, mut_rate=mut_rate,
                         max_depth=max_depth, n_jobs=n_jobs, verbose=verbose, random_state=random_state)

    def _add_estimators(self, pset, num_instances):
        classifiers.add_estimators(pset, num_instances)

    def _add_voters(self, pset):
        classifiers.add_voters(pset)

    def _to_callable(self, individual):
        # Currently need to do 2 evals. TODO: Reduce this to one
        init = self.toolbox.compile(expr=individual)
        return eval(str(init))