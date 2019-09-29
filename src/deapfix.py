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

    while len(stack) != 0:
        depth, type_ = stack.pop()
        if condition(height, depth):
            try:
                term = add_terminal(pset, type_)
                expr.append(term)
            except IndexError as e:
                # TODO: Can this checking be done in condition?

                # We couldnt find a terminal with this type, we want to keep growing in most cases till we get
                # a terminal node. This is one of the inconsistencies with strongly typed GP, we can not
                # just stop arbitrarily

                # In this case we need to keep growing, so add a primitive rather than a terminal
                if type_.__name__ == "ClassifierMixin":
                    # If we try add a classifier terminal, instead add a classifier primitive.
                    # Do not add a VotingClassifier  though or we risk getting stuck in an infinite loop
                    prim = next((x for x in pset.primitives[type_] if "Voting" not in x.name), None)
                else:
                    raise IndexError("No terminals found for type", type_, "please check function and terminal set")

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
