from deap import tools
import time


def random_search(population, toolbox, max_running_time=None, stats=None, halloffame=None, verbose=1):
    """"""
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # If the user specifies an early exit
    if max_running_time:
        end_time = time.time() + 60 * max_running_time

        if verbose:
            print("Running for", max_running_time, "minutes")

    evaluated = []

    for ind in population:

        if not ind.fitness.valid:
            fitness = toolbox.evaluate(ind)
            ind.fitness.values = fitness

        evaluated.append(ind)

        # Early exit if max time specified
        if max_running_time and time.time() >= end_time:

            if verbose:
                print("Time hit, exiting early.")

            break

    # Update our hall of fame to store the best individuals
    if halloffame is not None:
        halloffame.update(evaluated)

    # For tracking the stats of the pop
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(evaluated), **record)

    if verbose:
        print(logbook.stream)

    return population, logbook
