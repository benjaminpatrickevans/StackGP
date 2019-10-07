from src.base import Base

class StackGPClassifier(Base):

    def __init__(self, pop_size=100, max_run_time_mins=60, crs_rate=0.8, mut_rate=0.2, max_depth=17, n_jobs=1,
                 verbose=0, random_state=0):
        super().__init__(pop_size=pop_size, max_run_time_mins=max_run_time_mins, crs_rate=crs_rate, mut_rate=mut_rate,
                         max_depth=max_depth, n_jobs=n_jobs, verbose=verbose, random_state=random_state)
