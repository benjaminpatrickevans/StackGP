import random
import sys
from inspect import isclass
from deap import gp

######################################
# GP Program generation functions    #
######################################
def genGrow(pset, min_, max_, type_=None):
    """Generate an expression where each leaf might have a different depth
    between *min* and *max*.
    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A grown tree with leaves at possibly different depths.
    """
    def condition(height, depth):
        """Expression generation stops when the depth is equal to height
        or when it is randomly determined that a a node should be a terminal.
        """
        return depth >= height or \
            (depth >= min_ and random.random() < pset.terminalRatio)

    return generate(pset, min_, max_, condition, type_)


def genFull(pset, min_, max_, type_=None):
    """Generate an expression where each leaf has a the same depth
    between *min* and *max*.
    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A full tree with all leaves at the same depth.
    """
    def condition(height, depth):
        """Expression generation stops when the depth is equal to height."""
        return depth >= height
    return generate(pset, min_, max_, condition, type_)


def genHalfAndHalf(pset, min_, max_, type_=None):
    """Generate an expression with a PrimitiveSet *pset*.
    Half the time, the expression is generated with :func:`~deap.gp.genGrow`,
    the other half, the expression is generated with :func:`~deap.gp.genFull`.
    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: Either, a full or a grown tree.
    """
    method = random.choice((genGrow, genFull))
    return method(pset, min_, max_, type_)


def generate(pset, min_, max_, condition, type_=None):
    """Generate a Tree as a list of list. The tree is build
    from the root to the leaves, and it stop growing when the
    condition is fulfilled.
    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param condition: The condition is a function that takes two arguments,
                      the height of the tree to build and the current
                      depth in the tree.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A grown tree with leaves at possibly different depths
              dependending on the condition function.
    """
    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]

    should_stop_growing = False

    while len(stack) != 0:
        depth, type_ = stack.pop()
        if condition(height, depth) or should_stop_growing:
            try:
                term = add_terminal(pset, type_)
                expr.append(term)
            except IndexError as e:
                # We couldnt find a terminal with this type, we want to keep growing in most cases till we get
                # a terminal node. This is one of the inconsistencies with strongly typed GP, we can not
                # just stop arbitrarily

                # In this case we need to keep growing, so add a primitive rather than a terminal
                if type_.__name__ in ["ClassifierMixin", "RegressorMixin"]:

                    # If we try add a classifier terminal, instead add a classifier primitive.

                    # Do not add a VotingClassifier  though or we risk getting stuck in an infinite loop
                    bad_prefixes = ["Voting", "Stacking"]

                    # Find a primitive which does not begin with any of the prefixes in the bad_prefixes
                    allowed_primitives = [prim for prim in pset.primitives[type_]
                                if not any(prim.name.startswith(bad_prefix) for bad_prefix in bad_prefixes)]

                    prim = random.choice(allowed_primitives)

                    print("Using ", prim.name, " for a primitive since no terminal was found")

                    # TODO: We should set a flag to make sure we set dummy values for the preprocessing steps here
                    should_stop_growing = True

                    # Add the primitive and continue the loop
                    expr.append(prim)
                    for arg in reversed(prim.args):
                        stack.append((depth + 1, arg))

                else:
                    raise IndexError("No terminals found for type", type_, "please check function and terminal set")

        else:
            try:
                prim = random.choice(pset.primitives[type_])

                expr.append(prim)
                for arg in reversed(prim.args):
                    stack.append((depth + 1, arg))

            except IndexError:
                # CUSTOM
                # If we try add a primitive, and there is none available with this type_
                # instead try add a terminal of the same type. If theres still none available
                # only then will an exception will be thrown.
                term = add_terminal(pset, type_)
                expr.append(term)

    return expr


def add_terminal(pset, type_):
    try:
        term = random.choice(pset.terminals[type_])
    except IndexError:
        _, _, traceback = sys.exc_info()
        raise IndexError("The custom generate function tried to add " \
                         "a terminal of type '%s', but there is " \
                         "none available." % (type_,), traceback)
    if isclass(term):
        term = term()

    return term


def _unique_parents(population):
    """ Return two unique individuals from
    population if they exist, if not
    return a random individual and None

    :param population:
    :return: (ind1, ind2) where ind2 is None if no breedable parents found
    """
    # Breedable trees are at least one node high
    breedable_trees = [ind for ind in population if ind.height > 1]

    # If theres no breedable trees, then there's no point trying to crossover
    if not breedable_trees:
        return random.choice(population), None

    # At this stage there must be atleast one breadable tree
    parent_one = random.choice(breedable_trees)

    # See if we can find a unique breeder. Note: we didnt check breedabe_trees size >= 2, because
    # we could have duplicate individuals which we do not want to breed!
    unique_breeders = [ind for ind in breedable_trees if str(ind) != str(parent_one)]

    # If we cant breed, just return a random from the entire population
    if not unique_breeders:
        return random.choice(population), None

    # If we get to this point, it means we found two unique breeders
    return parent_one, random.choice(unique_breeders)


def varOr(population, toolbox, lambda_, cxpb, mutpb):
    """
    This is a variation the varOr function provided by deap
    (in algorithms.py).

    The difference is this tries to select 2 unique individuals
    when crossing over.
    """
    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")

    offspring = []
    for _ in range(lambda_):
        op_choice = random.random()
        if op_choice < cxpb:            # Apply crossover
            ind1, ind2 = map(toolbox.clone, _unique_parents(population))

            # If there was a second parent, we can do crossover
            if ind2:
                ind1, ind2 = toolbox.mate(ind1, ind2)
            else:
                # Otherwise we must have matching parents or parents only one node high.
                # So we should mutate instead to hopefully generate a new unique individual
                ind1, = toolbox.mutate(ind1)

            del ind1.fitness.values
            offspring.append(ind1)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = toolbox.clone(random.choice(population))
            ind, = toolbox.mutate(ind)
            del ind.fitness.values
            offspring.append(ind)
        else:                           # Apply reproduction
            offspring.append(random.choice(population))

    return offspring


def repeated_mutation(individual, expr, pset, existing, toolbox, max_tries=10):
    """
        Repeated apply mutUniform until the mutated individual has
        not existed before.
    :param individual:
    :param expr:
    :param pset:
    :return:
    """

    # Try for max_tries, or until we generate a unique individual
    for i in range(max_tries):
        ind = toolbox.clone(individual)

        mutated = gp.mutUniform(ind, expr, pset)

        # mutUniform returns a tuple, so access the first element of the tuple and see if that is unique
        if str(mutated[0]) not in existing:
            break

    return mutated


def repeated_crossover(ind1, ind2, existing, toolbox, max_tries=10):
    """
        Repeatedly apply cxOnePoint until the generated individuals are
        unique from the existing originals (or until max_tries is hit).
        Thiw was inspired by tpots _mate_operator.
    :param ind1:
    :param ind2:
    :param existing:
    :param toolbox:
    :param max_tries:
    :return:
    """
    unique_offspring1 = None
    unique_offspring2 = None

    # Try for max_tries, or until we generate a unique individual
    for i in range(max_tries):
        ind1_copy, ind2_copy = toolbox.clone(ind1), toolbox.clone(ind2)

        offspring1, offspring2 = gp.cxOnePoint(ind1_copy, ind2_copy)

        if str(offspring1) not in existing:
            unique_offspring1 = offspring1

        if str(offspring2) not in existing:
            unique_offspring2 = offspring2

        # Only break once both are unique
        if unique_offspring1 and unique_offspring2:
            break


    # If we didnt find a unique, then use the last (repeated) offspring generated
    unique_offspring1 = unique_offspring1 or offspring1
    unique_offspring2 = unique_offspring2 or offspring2

    return unique_offspring1, unique_offspring2
