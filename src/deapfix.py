import random
import sys
from inspect import isclass
from deap import gp

class SearchExhaustedException(Exception):
    pass

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
                term = add_terminal(pset, type_, should_stop_growing)
                expr.append(term)
            except IndexError as e:
                # We couldnt find a terminal with this type, we want to keep growing in most cases till we get
                # a terminal node. This is one of the inconsistencies with strongly typed GP, we can not
                # just stop arbitrarily
                should_stop_growing = True

                # In this case we need to keep growing, so add a primitive rather than a terminal
                if type_.__name__ in ["ClassifierMixin", "RegressorMixin"]:

                    # If we try add a classifier terminal, instead add a classifier primitive.

                    # Do not add a VotingClassifier  though or we risk getting stuck in an infinite loop
                    bad_prefixes = ["Voting", "Stacking"]

                    # Find a primitive which does not begin with any of the prefixes in the bad_prefixes
                    allowed_primitives = [prim for prim in pset.primitives[type_]
                                if not any(prim.name.startswith(bad_prefix) for bad_prefix in bad_prefixes)]

                else:
                    # Need to use the dummy nodes
                    allowed_primitives = [prim for prim in pset.primitives[type_] if prim.name.startswith("Dummy")]

                # If at this stage we havent found a replacement, we must raise an exception.
                if not allowed_primitives:
                    print("No suitable replacements found. This indicates a misconfiguration and should not occur")
                    raise e

                prim = random.choice(allowed_primitives)

                # Add the primitive and continue the loop
                expr.append(prim)
                for arg in reversed(prim.args):
                    stack.append((depth + 1, arg))

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


def add_terminal(pset, type_, requires_dummy=False):
    try:
        allowed_terminals = pset.terminals[type_]

        if requires_dummy and str(type_) in ["FeatureProcessorType", "DataProcessorType"]:
            # For STGP, if we are supposed to stop early then we add dummy terminals.
            # otherwise the effective branch length wont match whats desired
            allowed_terminals = [term for term in allowed_terminals if str(term.name).startswith("Dummy")]

        term = random.choice(allowed_terminals)
    except IndexError:
        _, _, traceback = sys.exc_info()
        raise IndexError("The custom generate function tried to add " \
                         "a terminal of type '%s', but there is " \
                         "none available." % (type_,), traceback)
    if isclass(term):
        term = term()

    return term


def _subtree_str(tree, starting_idx):
    """
    Returns the subtree from tree starting at
    starting_idx
    :param tree:
    :param starting_idx:
    :return:
    """
    subtree_slice = tree.searchSubtree(starting_idx)
    subtree = gp.PrimitiveTree(tree[subtree_slice])
    return str(subtree)


def _unique_parents_old(population):
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

    # See if we can find a unique breeder. Note: we didnt check breedable_trees size >= 2, because
    # we could have duplicate individuals which we do not want to breed!

    # A unique breeder must be different after the root. If the root is the same, but the children
    # match then we cant create a unique child with crossover so we do not consider these unique breeders
    parent_below_root = _subtree_str(parent_one, starting_idx=1)
    unique_breeders = [ind for ind in breedable_trees
                       if _subtree_str(ind, starting_idx=1) != parent_below_root]

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
            ind = diverse_crossover(population, toolbox)
            del ind.fitness.values
            offspring.append(ind)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = diverse_mutate(population, toolbox)
            del ind.fitness.values
            offspring.append(ind)
        else:                           # Apply reproduction
            offspring.append(random.choice(population))

    return offspring


def _unique_parents(population):
    # Breedable trees are at least one node high
    breedable_trees = [ind for ind in population if ind.height > 1]

    random.shuffle(breedable_trees)

    # See if we can find unique breeders in the population
    for parent in breedable_trees:

        # A unique breeder must be different after the root. If the root is the same, but the children
        # match then we cant create a unique child with crossover so we do not consider these unique breeders
        parent_below_root = _subtree_str(parent, starting_idx=1)
        unique_breeders = [ind for ind in breedable_trees
                           if _subtree_str(ind, starting_idx=1) != parent_below_root]

        for breeder in unique_breeders:
            yield parent, breeder

    # If we get to this stage, it means we couldnt find any unique breeders
    yield random.choice(population), None


def diverse_crossover(population, toolbox):
    """
        Finds two parents which when combined produce
        a unique individual which has not yet existed in the population.

    :param population:
    :param toolbox:
    :return:
    """

    for parent1, parent2 in _unique_parents(population):
        parent1 = toolbox.clone(parent1)
        parent2 = toolbox.clone(parent2)

        ind1, _ = toolbox.mate(parent1, parent2)

        # If there is an ind1, it means it was unique so return it
        if ind1:
            return ind1

    # If we get here we couldnt find a unique breeder, need to mutate one instead
    return diverse_mutate(population, toolbox)


def diverse_mutate(population, toolbox):
    """
        Mutates ind1 if it can create a unique individual (i.e. one that hasnt existed
        before). If not possible, finds a value from the population which can be
        mutated to generate unique individual.

    :param toolbox:
    :param ind1:
    :param population:
    :return:
    """
    shuffled_pop = shuffled(population)

    for individual in shuffled_pop:
        individual = toolbox.clone(individual)
        mutated_ind, _ = toolbox.mutate(individual)

        # If we were able to succesfully mutate the indiviual into a unique one
        if mutated_ind:
            return mutated_ind

    # At this stage we have tried the entire population and cant make a unique individual
    # This means the given search space has been entirely explored.
    print("Couldnt generate a unique individual! Exhausted the entire search space")
    raise SearchExhaustedException("Search space exhausted")

def uniqueMutUniform(individual, expr, pset, existing, toolbox):
    """
    Extension of gp.mutUniform which tries to mutate individual
    to one which hasnt existed before

    :param individual: The tree to be mutated.
    :param expr: A function object that can generate an expression when
                 called.
    :returns: A tuple of one tree.
    """
    indices = list(range(len(individual)))
    random.shuffle(indices)

    for index in indices:
        slice_ = individual.searchSubtree(index)
        type_ = individual[index].ret

        ind = toolbox.clone(individual)
        # TODO: We should try expr several times
        ind[slice_] = expr(pset=pset, type_=type_)

        if str(ind) not in existing:
            # Found a unique one, so return it
            return ind, None

    print("Mutate couldnt make unique individual")

    # Couldn't make a unique individual, so just return the original individual
    return individual, None


def uniqueCxOnePoint(ind1, ind2, existing, toolbox):
    """Randomly select in each individual and exchange each subtree with the
    point as root between each individual. Tries to ensure a unique crossover,
    i.e. attempts to generate a child which hasnt existed before.

    If a unique individual can not be generated, the return value will
    be (ind1, None).

    :param ind1: First tree participating in the crossover.
    :param ind2: Second tree participating in the crossover.
    :returns: A tuple of two trees.
    """
    if len(ind1) < 2 or len(ind2) < 2:
        # No crossover on single node tree
        return ind1, ind2

    # List all available primitive types in each individual
    types1 = gp.defaultdict(list)
    types2 = gp.defaultdict(list)

    for idx, node in enumerate(ind1[1:], 1):
        types1[node.ret].append(idx)
    for idx, node in enumerate(ind2[1:], 1):
        types2[node.ret].append(idx)
    common_types = set(types1.keys()).intersection(set(types2.keys()))

    # Cant crossover if no shared node types
    if not common_types:
        return ind1, None

    # Just to be safe, ensure its randomised so the loop below isnt biased to selecting a particular type
    common_types = list(common_types)
    random.shuffle(common_types)

    # Try find a crossover point which generates a tree which is different from its parents
    # Check all crossover points and exit if we find one
    for type_ in common_types:
        ind1_nodes = shuffled(types1[type_])
        ind2_nodes = shuffled(types2[type_])

        for node1 in ind1_nodes:

            slice1 = ind1.searchSubtree(node1)

            for node2 in ind2_nodes:
                slice2 = ind2.searchSubtree(node2)

                newind1 = toolbox.clone(ind1)
                newind2 = toolbox.clone(ind2)

                newind1[slice1], newind2[slice2] = ind2[slice2], ind1[slice1]

                newind1_str = str(newind1)
                newind2_str = str(newind2)

                # If the generated individual is different to its parent and never existed in a previous generation
                # TODO: Do we need the first check?
                if newind1_str != str(ind1) and newind1_str not in existing:
                    return newind1, newind2

                if newind2_str != str(ind2) and newind2_str not in existing:
                    return newind2, newind1

    # If we get to this point we havent found any suitable crossover points
    return None, None


def shuffled(l):
    random.shuffle(l)
    return l


def safeStaticLimit(key, max_value):
    """
    Extension of gp.staticLimit which will
    work with individuals which can be None
    """
    def decorator(func):
        @gp.wraps(func)
        def wrapper(*args, **kwargs):
            keep_inds = [gp.copy.deepcopy(ind) for ind in args]
            new_inds = list(func(*args, **kwargs))
            for i, ind in enumerate(new_inds):
                # All that was changed is the condition below to include a truthy check
                if ind and key(ind) > max_value:
                    new_inds[i] = random.choice(keep_inds)
            return new_inds
        return wrapper
    return decorator
