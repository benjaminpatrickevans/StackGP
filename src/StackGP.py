from src.base import Base
from sklearn.model_selection import cross_val_score
from src.required import *  # Needed for eval to recreate individuals, do not delete
import numpy as np


class StackGP(Base):

    def __init__(self, pop_size=100, generations=5, crs_rate=0.2, mut_rate=0.8, max_depth=17, verbose=0, random_state=0):
        super().__init__(pop_size=pop_size, generations=generations, crs_rate=crs_rate, mut_rate=mut_rate,
                         max_depth=max_depth, verbose=verbose, random_state=random_state)

    def _to_callable(self, individual):
        # Currently need to do 2 evals. TODO: Reduce this to one
        init = self.toolbox.compile(expr=individual)
        return eval(str(init))

    def _fitness_function(self, individual, x, y):
        tree_str = str(individual)

        # Avoid recomputing fitness
        if tree_str in self.cache:
            return self.cache[tree_str]

        pipeline = self._to_callable(individual)

        try:
            # Crossfold validation for fitness
            result = cross_val_score(pipeline, x, y, cv=3, scoring="f1_weighted")
            fitness = result.mean(),
        except ValueError as e:
            # Help with debugging, should prevent this from ever occuring
            print(e)

            # For example if select 0 features
            fitness = 0,

        # Store in cache so we don't need to reevaluate
        self.cache[tree_str] = fitness

        return fitness

    def predict(self, x):
        if self.model is None:
            raise Exception("Must call fit before predict")

        x = self.imputer.transform(x)

        if self.categorical_features:
            x = self._to_training_encoding(x)

        # Could be object type
        x = np.array(x, dtype=float)

        return self.model.predict(x)