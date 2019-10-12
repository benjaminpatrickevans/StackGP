from src import deapfix, search
import numpy as np
from deap import base, creator, tools, gp
import random
import operator
import time
from math import inf
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from src.components import CustomPipeline
from xgboost.core import XGBoostError

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

        self.pset = self.create_pset()
        self.toolbox = self.create_toolbox()

        self.model = None
        self.imputer = None
        self.one_hot_encoding = None

    def create_pset(self):
        # Takes in no parameters, and returns a estimator
        pset = gp.PrimitiveSetTyped("MAIN", [], self.est_type)

        # So we can recreate np arrays
        pset.context["array"] = np.array

        # So we can recreate sklearn pipelines
        pset.context["Pipeline"] = Pipeline
        pset.context["CustomPipeline"] = CustomPipeline

        return pset

    def create_toolbox(self):
        toolbox = base.Toolbox()

        # Maximising f1-score, minimising complexity
        creator.create('FitnessMulti', base.Fitness, weights=(1.0, -1.0))

        # Individuals are represented as trees, the typical GP representation
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti, pset=self.pset)

        # Between 1 layer and max depth high
        toolbox.register("expr", deapfix.genHalfAndHalf, pset=self.pset, min_=0, max_=3)

        # Crossover
        toolbox.register("mate", deapfix.repeated_crossover, existing=self.cache, toolbox=toolbox)

        # Mutation
        toolbox.register("expr_mut", deapfix.genHalfAndHalf, min_=0, max_=3)
        toolbox.register("mutate", deapfix.repeated_mutation, expr=toolbox.expr_mut, pset=self.pset, existing=self.cache,
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
        toolbox.register("compile", gp.compile, pset=self.pset)

        return toolbox

    @staticmethod
    def create_stats():
        stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values[0])
        stats_size = tools.Statistics(key=lambda ind: ind.fitness.values[1])

        mstats = tools.MultiStatistics(fitness=stats_fit, complexity=stats_size)

        mstats.register("min",  np.min)
        mstats.register("mean", np.mean)
        mstats.register("max", np.max)
        mstats.register("std", np.std)

        return mstats

    def fit(self, data_x, data_y, verbose=1):
        # Set evolutionary seed since GP is stochastic (reproducability)
        random.seed(self.random_state)
        np.random.seed(self.random_state)

        # Defined by the subclass
        self._add_components(self.pset)

        # Make it 1D
        data_y = data_y.values.reshape(-1,)

        # Register the fitness function, pass1ing in our training data for evaluation
        self.toolbox.register("evaluate", self._fitness_function, x=data_x, y=data_y)

        pop = self.toolbox.population(n=self.pop_size)
        stats = Base.create_stats()

        # For floating point numbers, define a "tolerance" for the pareto front
        similarity = lambda ind1, ind2: np.allclose(ind1.fitness.values, ind2.fitness.values)
        pareto_front = tools.ParetoFront(similar=similarity)

        pop, logbook, generations = search.eaTimedMuPlusLambda(population=pop, toolbox=self.toolbox, mu=self.pop_size,
                                   lambda_=self.pop_size, cxpb=self.crs_rate,
                                   mutpb=self.mut_rate,
                                   end_time=self.end_time, stats=stats, halloffame=pareto_front)

        if verbose:
            print("Best model found:", pareto_front[0], "with fitness of", pareto_front[0].fitness)
            print("Percentage of unique models", (len(self.cache) / (generations * self.pop_size)) * 100)

        # Use the model with the heighest fitness
        self.model = self._to_callable(pareto_front[0])
        self.model.fit(data_x, data_y)

        self._print_pareto(pareto_front)

        # Clear the cache to free memory now we have finished evolution
        self.cache = {}

    def _print_pareto(self, pareto_front):
            print([solution.fitness.values for solution in pareto_front])

    def _calculate_complexity(self, tree_str):
        # Complexity measured by the number of voting nodes - TODO: one pass
        complexity = (3 * tree_str.count("Voting3")) + (5 * tree_str.count("Voting5")) +\
                     (4 * tree_str.count("Stacking3")) + (6 * tree_str.count("Stacking5"))

        return complexity

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

        try:
            # Crossfold validation
            score = cross_val_score(model, x, y, cv=3, scoring=self.scorer, n_jobs=self.n_jobs)
            # Average across the folds
            score = score.mean()
        except ValueError as e:
            #raise e

            # TODO: Decide what to do in this case
            if self.verbose:
                print("Error occured in eval", e, "setting f1 score to 0")
            score = 0
        except XGBoostError as e:
            print("Error with xgboost", e)
            score = 0

        complexity = self._calculate_complexity(tree_str)

        # Fitness is the average score (across folds) and the complexity
        fitness = score, complexity

        # Store fitness in cache so we don't need to reevaluate
        self.cache[tree_str] = fitness

        return fitness

    def _to_callable(self, individual):
        # Currently need to do 2 evals. TODO: Reduce this to one
        init = self.toolbox.compile(expr=individual)
        return eval(str(init), self.pset.context, {})
