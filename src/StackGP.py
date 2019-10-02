from src.base import Base
from sklearn.model_selection import cross_val_score
from src.required import *  # Needed for eval to recreate individuals, do not delete
import time
from math import inf


class StackGP(Base):

    def __init__(self, pop_size=1024, max_run_time_mins=60, crs_rate=0.8, mut_rate=0.2, max_depth=17, n_jobs=1,
                 verbose=0, random_state=0):
        super().__init__(pop_size=pop_size, max_run_time_mins=max_run_time_mins, crs_rate=crs_rate, mut_rate=mut_rate,
                         max_depth=max_depth, n_jobs=n_jobs, verbose=verbose, random_state=random_state)

    def _to_callable(self, individual):
        # Currently need to do 2 evals. TODO: Reduce this to one
        init = self.toolbox.compile(expr=individual)
        return eval(str(init))

    def _calculate_complexity(self, tree_str):
        # Complexity measured by the number of voting nodes - TODO: one pass
        complexity = (3. * tree_str.count("Voting3")) + (5. * tree_str.count("Voting5"))

        # Max theoretical complexity would be if every internal node was a Voting5 node
        max_complexity = 5 ** self.max_depth

        return complexity / max_complexity

    def _fitness_function(self, individual, x, y):
        # Dont evaluate, we need to stop
        if time.time() > self.end_time:
            return -inf, inf

        tree_str = str(individual)

        # Avoid recomputing fitness
        if tree_str in self.cache:
            return self.cache[tree_str]

        model = self._to_callable(individual)

        # Crossfold validation
        f1 = cross_val_score(model, x, y, cv=3, scoring="f1_weighted", n_jobs=self.n_jobs)

        complexity = self._calculate_complexity(tree_str)

        # Fitness is the average f1 (across folds) and the complexity
        fitness = f1.mean(), complexity

        # Store fitness in cache so we don't need to reevaluate
        self.cache[tree_str] = fitness

        return fitness

    def predict(self, x):
        if self.model is None:
            raise Exception("Must call fit before predict")

        '''
        x = self.imputer.transform(x)

        if self.categorical_features:
            x = self._to_training_encoding(x)

        # Could be object type
        x = np.array(x, dtype=float)
        '''
        return self.model.predict(x)