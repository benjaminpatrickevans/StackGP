from deap import tools
import time
from functools import partial
import random
import numpy as np
from math import inf

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


def pairwise_tournament(individuals, k, toolbox):
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
    # Choose best individual from random pairs, k times. Clones the resulting individuals
    chosen = [toolbox.clone(_get_best(random.choice(individuals), random.choice(individuals)))
              for _ in range(k)]

    return chosen


def create_next_generation(generation_number, population, toolbox):
    # Choose half of the population to be parents. This is done as a 2 way tournament, with replacement.
    parents = pairwise_tournament(population, len(population) // 2, toolbox)

    # Mutate the parents to produce new child offsprings
    children = [toolbox.mutate(parent)[0] for parent in parents]

    # Do not inherit previous fitness from parents
    for child in children:
        child.previous_scores = None
        child.generation_created = generation_number

    # The next generation becomes the parents and children
    return parents + children


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
        # Need to compute fitness for entire population every generation, as the fitness function changes each step
        fitnesses = toolbox.map(partial(toolbox.evaluate, seed=gen), population)

        # Store the fitness. The first objective is averaged across all generations. When an individual is created from
        # mutation this will be inherited from the parents.
        for ind, current_fitness in zip(population, fitnesses):
            if ind.previous_scores is None:
                ind.previous_scores = []

            # Can be -inf if we timeout when evaluating fitness
            if current_fitness[0] != -inf:
                ind.previous_scores.append(current_fitness[0])

                # Use the average f1 score across all generations for first objective.
                # Average size doesnt make much sense. So just use the current size for second objective
                ind.fitness.values = np.mean(ind.previous_scores), current_fitness[1],

        # We need to clear the HOF each generation since the fitness function changes each gen
        hof.clear()
        hof.update(population)

        # Update the statistics for this generation
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, **record)
        if verbose:
            print(logbook.stream)

        # Now create the next generation
        population[:] = create_next_generation(gen + 1, population, toolbox)
        gen += 1

    return hof, logbook, gen



