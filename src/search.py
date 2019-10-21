from deap import tools, algorithms
import time
from functools import partial
import random
import numpy as np
from math import inf
from sklearn.metrics.pairwise import euclidean_distances


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

def varOr(population, toolbox, lambda_, cxpb, mutpb):
    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")

    offspring = []
    for _ in range(lambda_):
        op_choice = random.random()
        if op_choice < cxpb:            # Apply crossover
            ind1, ind2 = list(map(toolbox.clone, random.sample(population, 2)))
            ind1, ind2 = toolbox.mate(ind1, ind2)
            ind1.behaviour = None
            del ind1.fitness.values
            offspring.append(ind1)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = toolbox.clone(random.choice(population))
            ind, = toolbox.mutate(ind)
            del ind.fitness.values
            ind.behaviour = None
            offspring.append(ind)
        else:                           # Apply reproduction
            offspring.append(random.choice(population))

    return offspring

def novelty_search(population, toolbox, end_time, cxpb=0, mutpb=1, stats=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    offspring = population
    gen = 0
    K = 3  # Num neighbours

    # Store an archive of most novel individuals at the time of occurence
    archive_behaviour = []

    # Begin the generational process
    while time.time() < end_time:
        if archive_behaviour:
            print("Archive", len(archive_behaviour), len(archive_behaviour[0]))

        # Evaluate the individuals without a behaviour
        invalid_ind = [ind for ind in offspring if ind.behaviour is None]

        population_behaviour = toolbox.map(toolbox.behaviour, invalid_ind)
        population_behaviour = list(population_behaviour)

        all_behaviour = population_behaviour + archive_behaviour

        # At this point every individual has a behaviour, so we can calculate pairwise differences in behaviour
        # All pairwise differences for each predictor. Shape (#predictors, #predictors).
        behavioral_distance = euclidean_distances(all_behaviour)
        num_predictors = behavioral_distance.shape[1]

        # Sort the columns to be in terms of increasing distance for each row
        behavioral_distance.sort(axis=1)

        # Safety check to cap K to be the max number of neighbours
        K = min(num_predictors - 1, K)

        # Get the average distance to the K nearest predictors. Exclude the first one since thats the distance to self (0)
        average_distance_to_neighbours = behavioral_distance[:, 1:K + 1].mean(axis=1)

        #
        for ind, behaviour, novelty in zip(invalid_ind, population_behaviour, average_distance_to_neighbours):
            ind.behaviour = behaviour
            ind.fitness.values = novelty, 0

        archive_behaviour += population_behaviour

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Select the next generation individuals
        offspring = tools.selTournament(population, len(population), tournsize=2)

        # Vary the pool of individuals
        offspring = varOr(offspring, toolbox, len(population), cxpb, mutpb)
        gen += 1

    # Once done we now want to search all the resulting novel models for the best.
    candidates = population

    # Need to compute fitness for entire population every generation, as the fitness function changes each step
    fitnesses = toolbox.map(partial(toolbox.evaluate, timeout=False), candidates)

    # Store the fitness. The first objective is averaged across all generations.
    for ind, fitness in zip(candidates, fitnesses):
        ind.fitness.values = fitness

    # We only save the single best
    hof = tools.HallOfFame(1)
    hof.update(candidates)

    return hof, logbook, gen

def create_next_generation(generation_number, population, toolbox):
    # Choose half of the population to be parents. This is done as a 2 way tournament, with replacement.
    parents = pairwise_tournament(population, len(population) // 2, toolbox)

    # Mutate the parents to produce new child offsprings
    children = [toolbox.mutate(parent)[0] for parent in parents]

    # Do not inherit previous fitness from parents
    for child in children:
        child.previous_scores = None
        child.generation_created = generation_number

    # The next generation becomes the children and the parents
    return children + parents


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

        # Store the fitness. The first objective is averaged across all generations.
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



