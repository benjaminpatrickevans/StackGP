from deap import tools, gp
from src import customdeap
import time
from src.customdeap import SearchExhaustedException
from functools import partial
from hyperopt import fmin, tpe, hp, space_eval
from sklearn.model_selection import cross_val_score
from collections import OrderedDict

def eaTimedMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, end_time, pset,
                        stats=None, halloffame=None, verbose=__debug__):
    """
        This is the :math:`(\mu + \lambda)` evolutionary algorithm.
        This is a modification of the DEAP version: eaMuPlusLambda,
        with the only diference being running for max_time rather
        than ngen.
    """

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    offspring = population[:]
    gen = 0

    while time.time() < end_time:
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            bayesian_parameter_optimisation(ind, toolbox, [], [], pset)
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        gen += 1

        try:
            # Vary the population for next generation
            offspring = customdeap.varOr(population, toolbox, lambda_, cxpb, mutpb)
        except SearchExhaustedException:
            print("Search finished, exiting early")
            # If this happens we have exhausted our search space
            break

    return population, logbook, gen


def _fill_with_hyperparameters(tree, toolbox, params):
    """
    Creates a copy of the tree and changes the the parameters
    to the ones specified in params.
    :param tree:
    :param toolbox:
    :param params: A dict from node idx -> parameter value
    :return:
    """
    tree = toolbox.clone(tree)

    for idx, value in params.items():
        existing_node = tree[int(idx)]
        new_value = existing_node.ret(value)  # Need to wrap it in the original type for STGP

        # Replace the value with the updated one
        existing_node.value = new_value

    return tree

def objective_function(tree, valid_x, valid_y, toolbox, *params):
    """
    What we would like to optimise. In this case we want to maximise the f1 score
    :return:

    """
    tree = _fill_with_hyperparameters(tree, toolbox, *params)

    return 1

    # TODO: Use seperate set for bayesian
    f1 = cross_val_score(tree, valid_x, valid_y, cv=3, scoring="f1_weighted").mean()

    # Hyperopt minimises
    return -f1


def bayesian_parameter_optimisation(tree, toolbox, valid_x, valid_y, pset):
    """
    Optimises the parameters in tree with bayesian optimisation.
    Returns a copy of the tree with the updated hyperparameters.
    :param tree:
    :return:
    """
    # Need to convert the tree into a function which takes the hyperparameters as arguments
    hyperparameters = {idx: hp.choice(str(idx), node.value.range) for idx, node in enumerate(tree)
                 if isinstance(node, gp.Terminal) and hasattr(node.value, "hyper_parameter")}

    if not hyperparameters:
        # Cant optimise a tree with no tunable args, so just return a copy of the original tree
        return toolbox.clone(tree)

    best = fmin(
        fn=partial(objective_function, tree, valid_x, valid_y, toolbox),
        space=hyperparameters,
        algo=tpe.suggest,
        max_evals=10,
    )

    optimised_params = space_eval(hyperparameters, best)

    return _fill_with_hyperparameters(tree, toolbox, optimised_params)
