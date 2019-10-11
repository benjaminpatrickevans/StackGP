from src.base import Base
from src import components, regressors
from sklearn.base import RegressorMixin as Regressor

class StackGPRegressor(Base):

    def __init__(self, pop_size=100, max_run_time_mins=60, crs_rate=0.8, mut_rate=0.2, max_depth=17, n_jobs=1,
                 verbose=0, random_state=0):
        self.est_type = Regressor
        self.scorer = "neg_mean_squared_error"

        super().__init__(pop_size=pop_size, max_run_time_mins=max_run_time_mins, crs_rate=crs_rate, mut_rate=mut_rate,
                         max_depth=max_depth, n_jobs=n_jobs, verbose=verbose, random_state=random_state)

    def _add_components(self, pset):
        components.add_estimators(pset, regressors.estimators, regressors.Regressor)
        regressors.add_voters(pset)