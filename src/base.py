from src import deapfix, customtypes, components, search, data_helper, sklearn_additions
import numpy as np
import pandas as pd
from deap import base, creator, tools, gp
from sklearn.utils import shuffle
from sklearn.base import ClassifierMixin as Classifier
import inspect
import random
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class Base:
    """
        This class implements all the basic GP
        code needed to evolve a population.
        This should not be instantiated directly,
        rather use a base class.
    """

    def __init__(self, pop_size, max_running_time, verbose, random_state):

        self.pop_size = pop_size
        self.max_running_time = max_running_time
        self.verbose = verbose
        self.random_state = random_state

        self.pset = Base.create_pset()
        self.toolbox = Base.create_toolbox(self.pset)

        self.model = None
        self.encoder = None
        self.imputer = None
        self.one_hot_encoding = None

    @staticmethod
    def create_pset():
        # Takes in no parameters, and returns a classifier
        pset = gp.PrimitiveSetTyped("MAIN", [], Classifier)

        # We need to add all our custom defined types so they can be recreated
        for name, defined_type in inspect.getmembers(customtypes):
            if type(defined_type) == type:
                pset.context[name] = defined_type

        return pset

    @staticmethod
    def create_toolbox(pset):
        toolbox = base.Toolbox()

        # Fitness is f1-score. So the larger the better
        creator.create("FitnessMax", base.Fitness, weights=(1.0, ))

        # Individuals are represented as trees, the typical GP representation
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=pset)

        # Between 1 node and 3 high
        toolbox.register("expr", deapfix.genHalfAndHalf, pset=pset, min_=1, max_=3)

        # Individuals should be made based on the expr method above
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)

        # The population is just a list of individuals
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Use the standard deap method of compiling an individual
        toolbox.register("compile", gp.compile, pset=pset)

        return toolbox

    @staticmethod
    def create_stats():
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", np.max)
        return stats

    def _fill_missing(self, x):
        self.imputer = sklearn_additions.DataFrameImputer()
        return  self.imputer.fit_transform(x)

    def _to_training_encoding(self, x):
        '''
            http://fastml.com/how-to-use-pd-dot-get-dummies-with-the-test-set/
        :param x:
        :return:
        '''

        categorical_features = self._categorical_features(x)

        x = pd.get_dummies(x, columns=categorical_features)

        # In case we didnt see some of the examples in the test set
        missing_cols = set(self.training_columns) - set(x.columns)
        for c in missing_cols:
            x[c] = 0

        # Make sure we have all the columns we need
        assert (set(self.training_columns) - set(x.columns) == set())

        extra_cols = set(x.columns) - set(self.training_columns)

        if extra_cols:
            print("Extra columns in the unseen test data:", extra_cols)
            print("Ignoring them.")

        # Reorder to ensure we have the same columns and ordering as training data
        x = x[self.training_columns]

        return x

    def is_number(self, x):
        try:
            float(x)
            return True
        except ValueError:
            return False

    def _categorical_features(self, data_x):
        '''
            Returns the categorical column names
            from data_x. Returns an empty set
            if they are all numeric.
        :param data_x:
        :return:
        '''

        categorical_features = []

        for col in data_x:
            values = np.unique(data_x[col].values)

            # If they are all numeric
            numeric = np.all([self.is_number(x) for x in values])

            if not numeric:
                categorical_features.append(col)

        return categorical_features


    def fit(self, data_x, data_y, verbose=1):

        # Set evolutionary seed since GP is stochastic (reproducability)
        random.seed(self.random_state)
        np.random.seed(self.random_state)

        # First thing we need to do is deal with missing values
        data_x = self._fill_missing(data_x)

        # Cateogircal features are those where all the values are not numeric
        self.categorical_features = self._categorical_features(data_x)

        # Convert any categorical values to numeric
        data_x = pd.get_dummies(data_x, columns=self.categorical_features)
        self.training_columns = data_x.columns # For recreating at test time

        if self.categorical_features:
            print("Applied one hot encoding", len(self.training_columns))
            self.one_hot_encoding = True

        # Now we can just treat as floats since no categories left
        data_x = np.array(data_x, dtype=float)

        # Also we want to be safe and ensure y values are numpy
        data_y = np.asarray(data_y, dtype=str).flatten()

        # Some methods (i.e. chi-2 are only relevant when features are all positive.
        # Note: this will break if the testing set has negative values and the training doesnt
        all_non_negative_values = np.all(data_x >= 0)

        num_instances, num_features = data_x.shape
        num_classes = len(np.unique(data_y))

        components.add_classifiers(self.pset, num_instances, num_features, num_classes)
        components.add_combiners(self.pset)

        # Register the fitness function, passing in our training data for evaluation
        self.toolbox.register("evaluate", self._fitness_function, x=data_x, y=data_y)

        pop = self.toolbox.population(n=self.pop_size)
        stats = Base.create_stats()

        hof = tools.HallOfFame(1) # Only care for the single best found

        search.random_search(pop, self.toolbox, max_running_time=self.max_running_time, stats=stats, halloffame=hof)

        if verbose:
            print("Best model found:", hof[0])

        # Use the model with the heighes fitness
        self.model = self._to_callable(hof[0])
        self.model.fit(data_x, data_y)