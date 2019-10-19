from src import customdeap, search, scorer, viz
import numpy as np
from deap import base, creator, tools, gp
import random
import operator
import time
from math import inf
from sklearn.pipeline import Pipeline
from src.components import CustomPipeline


class Base:
    """
        This class implements all the basic GP
        code needed to evolve a population.
        This should not be instantiated directly,
        rather use a base class.
    """

    def __init__(self, pop_size, max_run_time_mins, max_eval_time_mins, max_depth, n_jobs, verbose, random_state):

        self.pop_size = pop_size
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

        self.max_run_time_mins = max_run_time_mins
        self.max_eval_time_mins = max_eval_time_mins

        # For generating unique models
        self.cache = {}

        self.pset = self.create_pset()
        self.toolbox = self.create_toolbox()

        self.callable_model = None
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

        creator.create('FitnessMulti', base.Fitness, weights=(1.0, -1.0))

        # Individuals are represented as trees, the typical GP representation
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti, pset=self.pset, previous_scores=None)

        # Between 1 layer and max depth high
        toolbox.register("expr", customdeap.genHalfAndHalf, pset=self.pset, min_=0, max_=3)

        # Individuals should be made based on the expr method above
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)

        # Mutation
        toolbox.register("expr_mut", customdeap.genHalfAndHalf, min_=0, max_=3)
        toolbox.register("mutate", customdeap.mutate_choice, pset=self.pset, expr=toolbox.expr_mut,
                         existing=self.cache, toolbox=toolbox)
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_depth))

        # The population is just a list of individuals
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Compile is used to turn a tree into runnable python code
        toolbox.register("compile", self.compile, pset=self.pset)

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
        # How long can we run fit for in seconds
        self.end_time = time.time() + (self.max_run_time_mins * 60)

        num_instances, num_features = data_x.shape

        # Set evolutionary seed since GP is stochastic (reproducability)
        random.seed(self.random_state)
        np.random.seed(self.random_state)

        # Defined by the subclass
        self._add_components(self.pset, num_features)

        # Make the y labels 1D
        data_y = data_y.values.reshape(-1,)

        # Register the fitness function, pass1ing in our training data for evaluation
        self.toolbox.register("evaluate", self._fitness_function, x=data_x, y=data_y)

        pop = self.toolbox.population(n=self.pop_size)
        stats = Base.create_stats()
        hof = tools.HallOfFame(1)

        hof, self.logbook, generations =\
            search.elitist_mutations(population=pop, toolbox=self.toolbox, end_time=self.end_time,
                                     stats=stats, verbose=self.verbose)

        #hof, self.logbook, generations =\
        #    search.eaTimedMuPlusLambda(population=pop, toolbox=self.toolbox, mu=self.pop_size, lambda_=self.pop_size,
        #                               end_time=self.end_time, stats=stats)

        if verbose:
            print("Best model found:", hof[0], "with fitness of", hof[0].fitness)
            print("Percentage of unique models:", (len(self.cache) / (generations * self.pop_size)) * 100)

        # Use the model with the heighest fitness
        self.model = hof[0]
        self.callable_model = self.toolbox.compile(hof[0])
        self.callable_model.fit(data_x, data_y)

        print("Pareto front:", [solution.fitness.values for solution in hof])

        # Clear the cache to free memory now we have finished evolution
        self.cache = {}


    def _calculate_complexity(self, tree):
        # Complexity measured as the number of estimators in a model.
        complexity = sum([1 for node in tree if node.ret == self.est_type])

        return complexity

    def predict(self, x):
        if self.callable_model is None:
            raise Exception("Must call fit before predict")

        return self.callable_model.predict(x)

    def _fitness_function(self, individual, x, y, seed=0, timeout=True, save_in_cache=False):
        seconds_left = self.end_time - time.time()

        # Dont evaluate, we need to stop as the time is up
        if timeout and seconds_left <= 0:
            return -inf, inf

        tree_str = str(individual)

        # Avoid recomputing fitness
        if tree_str in self.cache:
            return self.cache[tree_str]

        model = self.toolbox.compile(individual)

        try:
            # Time out if we pass the allowed amount of time.
            allowed_time_seconds = int(min(seconds_left, self.max_eval_time_mins * 60))
            score = scorer.timed_cross_validation(model, x, y, self.scoring_fn, allowed_time_seconds,
                                                  n_jobs=self.n_jobs, num_folds=3, seed=seed)
        except Exception as e:
            # This can occur if the individual model throws an exception, or if the function times out
            if self.verbose:
                print("Error occured in eval", e, "setting f1 score to 0")
                print("Tree was", tree_str)
            score = 0

        complexity = self._calculate_complexity(individual)

        # Fitness is the average score (across folds) and the complexity
        fitness = score, complexity

        if save_in_cache:
            # Store fitness in cache so we don't need to reevaluate
            self.cache[tree_str] = fitness

        return fitness

    def compile(self, individual, pset):
        # Currently need to do 2 evals. TODO: Reduce this to one
        init = gp.compile(expr=individual, pset=pset)
        return eval(str(init), self.pset.context, {})

    def plot(self, file_name):
        viz.plot_tree(self.model, file_name)

    def get_generation_information(self):
        fit_mins = self.logbook.chapters["fitness"].select("max")
        size_avgs = self.logbook.chapters["complexity"].select("mean")

        return {"fitness": fit_mins, "complexity": size_avgs}
