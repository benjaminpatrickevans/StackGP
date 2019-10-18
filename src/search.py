from deap import tools, gp
from src import customdeap
import time
from src.customdeap import SearchExhaustedException
from functools import partial
from hyperopt import fmin, tpe, hp, space_eval, Trials
from hyperopt.fmin import generate_trials_to_calculate
from sklearn.model_selection import cross_val_score
import numpy as np

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


def _fill_with_hyperparameters(tree, toolbox, hyperparameter_indices, params):
    tree = toolbox.clone(tree)

    for i, param_idx in enumerate(hyperparameter_indices):
        existing_node = tree[param_idx]

        value = params[i]

        new_value = existing_node.ret(value)  # Need to wrap it in the original type for STGP

        # Replace the value with the updated one
        existing_node.value = new_value

    return tree

def objective_function(tree, valid_x, valid_y, toolbox, hyperparameter_indices, *params):
    """
    What we would like to optimise. In this case we want to maximise the f1 score
    :return:

    """
    tree = _fill_with_hyperparameters(tree, toolbox, hyperparameter_indices, *params)

    return 1

    # TODO: Use seperate set for bayesian
    f1 = cross_val_score(tree, valid_x, valid_y, cv=3, scoring="f1_weighted").mean()

    # Hyperopt minimises
    return -f1

def _determine_sampling_type(node_values):
    return hp.choice

def get_hyperparameters_from_tree(tree):
    hyperparameters = []
    hyperparameter_indices = []
    defaults = {}

    for idx, node in enumerate(tree):
        if isinstance(node, gp.Terminal) and hasattr(node.value, "hyper_parameter"):
            hyperparameter_name = str(idx)

            node_values = node.value.range
            sampler = _determine_sampling_type(node_values)
            hyperparameters.append(sampler(str(idx), node_values))
            hyperparameter_indices.append(idx)

            default = node.value.val

            # Weird behaviours with hyeropt which expects choice to take an index, not the value
            if sampler.__name__ == "hp_choice":
                default = node_values.index(default)

            defaults[str(idx)] = default

    return hyperparameters, hyperparameter_indices, defaults

def bayesian_parameter_optimisation(tree, toolbox, valid_x, valid_y, pset):
    """
    Optimises the parameters in tree with bayesian optimisation.
    Returns a copy of the tree with the updated hyperparameters.
    :param tree:
    :return:
    """
    hyperparameters, hyperparameter_indices, default_values = get_hyperparameters_from_tree(tree)

    if not hyperparameters:
        # Cant optimise a tree with no tunable args, so just return a copy of the original tree
        return toolbox.clone(tree)

    # Start the search at the existing values rather than randomly
    trials = generate_trials_to_calculate([default_values])

    best = fmin(
        fn=partial(objective_function, tree, valid_x, valid_y, toolbox, hyperparameter_indices),
        space=hyperparameters,
        algo=tpe.suggest,
        max_evals=20,
        trials=trials
    )

    optimised_params = space_eval(hyperparameters, best)
    tree = _fill_with_hyperparameters(tree, toolbox, hyperparameter_indices, optimised_params)

    return tree
