from deap import tools
from src import customdeap
import time
from functools import partial
import random
import numpy as np


def _get_best(ind1, ind2):
    # Choose the ind with the highest fitness. Break ties by selecting one with lowest complexity
    if ind1.fitness.values[0] == ind2.fitness.values[0]:
        # Tie. Choose lowest complexity
        if ind1.fitness.values[1] <= ind2.fitness.values[1]:
            return ind1
        else:
            return ind2
    elif ind1.fitness.values[0] > ind2.fitness.values[0]:
        return ind1
    else:
        return ind2


def pairwise_tournament(individuals, k, toolbox, mutate=False):
    """Select the best individual among randomly chosen
    pairs, *k* times. Ties are split
    based on the complexity. The list returned contains
    copies of the selected individuals.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param mutate: If set to true the best of the pair will be mutated
    :returns: A list of selected (and potentially mutated) individuals.

    This function uses the :func:`~random.choice` function from the python base
    :mod:`random` module.
    """
    chosen = []

    for i in range(k):
        ind1 = random.choice(individuals)
        ind2 = random.choice(individuals)
        best = _get_best(ind1, ind2)

        # Clone so we dont accidentally modify ind1 or ind2
        choice = toolbox.clone(best)

        if mutate:
            choice, = toolbox.mutate(best)
            del choice.fitness.values

        # Since we can have duplicates we should clone to avoid issues
        chosen.append(choice)

    return chosen


def elitist_mutations(population, toolbox, end_time, stats=None, verbose=__debug__):
    """
        This is a modification on the algorithm proposed in "Large-Scale Evolution of Image Classifiers"
        https://arxiv.org/abs/1703.01041
    """
    # We only save the single best
    hof = tools.HallOfFame(1)

    logbook = tools.Logbook()
    logbook.header = ['gen'] + (stats.fields if stats else [])

    gen = 0

    while time.time() < end_time:
        # Need to compute fitness for entire offspring every generation, as the fitness function changes each step
        fitnesses = toolbox.map(partial(toolbox.evaluate, seed=gen), population)

        # Store the fitness. The first objective is averaged across all generations. When an individual is created from
        # mutation this will be inherited from the parents.
        for ind, current_fitness in zip(population, fitnesses):
            if ind.previous_scores is None:
                ind.previous_scores = []

            ind.previous_scores.append(current_fitness[0])

            # Use the average f1 score across all generations for first objective.
            # Average size doesnt make much sense. So just use the current size for second objective
            ind.fitness.values = np.mean(ind.previous_scores), current_fitness[1],

        # Choose half of the population to be parents. This is done as a 2 way tournament, with replacement.
        parents = pairwise_tournament(population, len(population) // 2, toolbox)

        # Mutate the parents to produce new child offsprings
        children = [toolbox.mutate(parent)[0] for parent in parents]

        # The next generation becomes the parents and children
        population[:] = parents + children

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, **record)
        if verbose:
            print(logbook.stream)

        gen += 1

    # Update the hall of fame with the generated individuals. We do this outside the loop
    # since the fitness function changes each evolution.
    hof.update(population)

    return hof, logbook, gen


def eaTimedMuPlusLambda(population, toolbox, mu, lambda_, end_time, stats=None, verbose=__debug__):
    """
        This is the :math:`(\mu + \lambda)` evolutionary algorithm.
        This is a modification of the DEAP version: eaMuPlusLambda,
        with the only diference being running for max_time rather
        than ngen.
    """

    # For floating point numbers, define a "tolerance" for the pareto front
    similarity = lambda ind1, ind2: np.allclose(ind1.fitness.values, ind2.fitness.values)
    hof = tools.ParetoFront(similar=similarity)

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    offspring = population[:]
    gen = 0

    while time.time() < end_time:
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        # Cache the results of the evaluation
        fitnesses = toolbox.map(partial(toolbox.evaluate, save_in_cache=True), invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        hof.update(offspring)

        # Select the next generation population using NSGA
        population[:] = tools.selNSGA2(population + offspring, mu, nd="log")

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        gen += 1

        # The off spring is the result of mutating best pairs
        offspring = pairwise_tournament(population, lambda_, toolbox, mutate=True)

    return hof, logbook, gen




