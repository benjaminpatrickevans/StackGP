from deap import tools
from src import deapfix
import time
from src.deapfix import SearchExhaustedException

def eaTimedMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, end_time,
                        stats=None, halloffame=None, verbose=__debug__):
    """
        This is the :math:`(\mu + \lambda)` evolutionary algorithm.
        This is a modification of the DEAP version: eaMuPlusLambda,
        with the only diference being running for max_time rather
        than ngen.
    """

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    offspring = population
    gen = 0

    while time.time() < end_time:
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
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
            offspring = deapfix.varOr(population, toolbox, lambda_, cxpb, mutpb)
        except SearchExhaustedException:
            print("Search finished, exiting early")
            # If this happens we have exhausted our search space
            break

    return population, logbook, gen
