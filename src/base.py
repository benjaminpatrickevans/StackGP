from src import deapfix, search
import numpy as np
from deap import base, creator, tools, gp
from sklearn.base import ClassifierMixin as Classifier
import inspect
import random
import operator
import time
from math import inf
from sklearn.model_selection import cross_val_score


class Base:
    """
        This class implements all the basic GP
        code needed to evolve a population.
        This should not be instantiated directly,
        rather use a base class.
    """

    def __init__(self, pop_size, max_run_time_mins, crs_rate, mut_rate, max_depth, n_jobs, verbose, random_state):

        self.pop_size = pop_size
        self.crs_rate = crs_rate
        self.mut_rate = mut_rate
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

        self.end_time = time.time() + (max_run_time_mins * 60)

        # For generating unique models
        self.cache = {}

        self.pset = Base.create_pset()
        self.toolbox = self.create_toolbox(self.pset)

        self.model = None
        self.imputer = None
        self.one_hot_encoding = None

    @staticmethod
    def create_pset():
        # Takes in no parameters, and returns a classifier
        pset = gp.PrimitiveSetTyped("MAIN", [], Classifier)

        return pset

    def create_toolbox(self, pset):
        toolbox = base.Toolbox()

        # Maximising f1-score, minimising complexity
        creator.create('FitnessMulti', base.Fitness, weights=(1.0, -1.0))

        # Individuals are represented as trees, the typical GP representation
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti, pset=pset)

        # Between 1 layer and max depth high
        toolbox.register("expr", deapfix.genHalfAndHalf, pset=pset, min_=0, max_=self.max_depth)

        # Crossover
        toolbox.register("mate", deapfix.repeated_crossover, existing=self.cache, toolbox=toolbox)

        # Mutation
        toolbox.register("expr_mut", deapfix.genHalfAndHalf, min_=0, max_=self.max_depth)
        toolbox.register("mutate", deapfix.repeated_mutation, expr=toolbox.expr_mut, pset=pset, existing=self.cache,
                         toolbox=toolbox)

        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_depth))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_depth))

        # Selection
        toolbox.register("select", tools.selNSGA2, nd="log")

        # Individuals should be made based on the expr method above
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)

        # The population is just a list of individuals
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Use the standard deap method of compiling an individual
        toolbox.register("compile", gp.compile, pset=pset)

        return toolbox

    @staticmethod
    def create_stats():
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("min",  np.min)
        stats.register("mean", np.mean)
        stats.register("max", np.max)
        stats.register("std", np.std)

        return stats

    def fit(self, data_x, data_y, verbose=1):
        # Set evolutionary seed since GP is stochastic (reproducability)
        random.seed(self.random_state)
        np.random.seed(self.random_state)

        num_instances, num_features = data_x.shape

        self._add_estimators(self.pset, num_instances)
        self._add_voters(self.pset)

        # Register the fitness function, passing in our training data for evaluation
        self.toolbox.register("evaluate", self._fitness_function, x=data_x, y=data_y)

        pop = self.toolbox.population(n=self.pop_size)
        stats = Base.create_stats()

        pareto_front = tools.ParetoFront()

        pop, logbook, generations = search.eaTimedMuPlusLambda(population=pop, toolbox=self.toolbox, mu=self.pop_size,
                                   lambda_=self.pop_size, cxpb=self.crs_rate,
                                   mutpb=self.mut_rate,
                                   end_time=self.end_time, stats=stats, halloffame=pareto_front)

        if verbose:
            print("Best model found:", pareto_front[0])
            print("Percentage of unique models", (len(self.cache) / (generations * self.pop_size)) * 100)

        # Use the model with the heighest fitness
        self.model = self._to_callable(pareto_front[0])
        self.model.fit(data_x, data_y)

        # Clear the cache to free memory now we have finished evolution
        self.cache = {}



    def _calculate_complexity(self, tree_str):
        # Complexity measured by the number of voting nodes - TODO: one pass
        complexity = (3. * tree_str.count("Voting3")) + (5. * tree_str.count("Voting5"))

        # Max theoretical complexity would be if every internal node was a Voting5 node
        max_complexity = 5 ** self.max_depth

        return complexity / max_complexity

    def predict(self, x):
        if self.model is None:
            raise Exception("Must call fit before predict")

        return self.model.predict(x)

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
