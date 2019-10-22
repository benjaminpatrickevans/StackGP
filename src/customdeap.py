import random
import sys
from inspect import isclass
from deap import gp
from collections import defaultdict
from copy import copy
from src.bayesopt import bayesian_parameter_optimisation


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
                    # Find a primitive that will not cause a loop, for example not keep trying to add voting nodes.
                    allowed_primitives = [prim for prim in pset.primitives[type_]
                                          if prim.ret not in prim.args]
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


def _get_children_indices(node, subtree):
    # Keep track of the children we've already replaced so we dont duplicate branches
    child_idx = 1  # First child is at index 1 of the tree
    node_to_replace_child_indices = defaultdict(list)  # Map from child_type -> [indices]

    # Get the indices of all the children in the original subtree
    for child_type in node.args:
        # Slice_ will be slice(child subtree root index, child subtree final node)
        slice_ = subtree.searchSubtree(child_idx)
        node_to_replace_child_indices[child_type].append(child_idx)

        # Next child is after this childs subtree
        child_idx = slice_.stop

    return node_to_replace_child_indices


def mutate_choice(individual, pset, expr, toolbox):
    methods = [mutShrink, mutNodeReplacement, mutUniform, mutInsert, mutBayesian]

    # Try each mutation method until we get an individual different to the input
    for method in shuffled(methods):
        ind = method(individual, pset, expr, toolbox)

        if str(ind) != str(individual):
            return ind,

    # At this point we couldnt mutate to something different than the original
    return toolbox.clone(individual),


def mutShrink(individual, pset, expr, toolbox):
    """This operator shrinks the *individual* by choosing a random voting node
    and randomly replacing it with one of its children.

    :param individual: The tree to be shrinked.
    :returns: A tuple of one tree.
    """
    # Cant shrink a stump
    if individual.height <= 1:
        return None, None

    iprims = []
    for i, node in enumerate(individual[1:], 1):
        # A shrinkable node is one which returns the same type as one of its children
        if isinstance(node, gp.Primitive) and node.ret in node.args:
            iprims.append((i, node))

    # For each possible shrinkable primitive
    for index, prim in shuffled(iprims):
        # Find the children which could replace this node (i.e. have a child with same return type as prim)
        replacement_children = [i for i, type_ in enumerate(prim.args) if type_ == prim.ret]

        # For the possible children, check if shrinking them would make a unique tree
        for child in shuffled(replacement_children):
            rindex = index + 1
            for _ in range(child + 1):
                rslice = individual.searchSubtree(rindex)
                subtree = individual[rslice]
                rindex += len(subtree)

            slice_ = individual.searchSubtree(index)
            ind = toolbox.clone(individual)
            ind[slice_] = subtree

            if str(ind) != str(individual):
                return ind

    # Couldnt generate a unique individual. Must have been no shrinkable params
    return toolbox.clone(individual)


def mutNodeReplacement(individual, pset, expr, toolbox):
    """Replaces a randomly chosen primitive from *individual* by a randomly
    chosen primitive with the same return type. Only the shared children
    types are transferred.

    :param individual: The normal or typed tree to be mutated.
    :returns: A tuple of one tree.
    """

    indices = list(range(len(individual)))

    # For each possible node in the tree
    for index in shuffled(indices):
        # See if we can mutate the node to create unique individual
        node_to_replace = individual[index]
        subtree_to_replace = gp.PrimitiveTree(individual[individual.searchSubtree(index)])

        if node_to_replace.arity == 0:  # Terminal
            # For a terminal theres no children so we can just do a straight replacement
            replacements = [node for node in pset.terminals[node_to_replace.ret]
                            if node.name != node_to_replace.name]

            # See if any of the replacements will generate a unique tree
            for term in shuffled(replacements):
                if isclass(term):
                    term = term()

                ind = toolbox.clone(individual)
                ind[index] = term

                if str(ind) != str(individual):
                    return ind

        else:   # Primitive
            # If the node has children, then we will swap the common children and generate new children for whats missing
            node_to_replace_children_types = set(node_to_replace.args)

            # For a primitive to replace this one, it must have a shared child type
            prims = [p for p in pset.primitives[node_to_replace.ret]
                     if node_to_replace_children_types.intersection(p.args) and p.name != node_to_replace.name]

            if not prims:
                # If this happens, then there where no shared children. So lets just get
                # any node and we will have to generate all the children instead. This
                # only happens if we are at the bottom of a pipeline (a data processor)
                # and that data processor has hyperparameters set. In this case we will just swap
                # out the processor for another and generate new hyperparameters.
                # TODO: We should be able to select from terminals here too
                prims = [p for p in pset.primitives[node_to_replace.ret] if p.name != node_to_replace.name]

            node_to_replace_child_indices = _get_children_indices(node_to_replace, subtree_to_replace)

            # See if any of the replacement nodes will generate a unique tree
            for replacement_node in shuffled(prims):
                # Make a copy of the child indices since we modify this as we iterate
                node_to_replace_children = copy(node_to_replace_child_indices)

                # We will build this tree up in the order a depth-first search would return
                replacement_tree = [replacement_node]

                # Now lets create the tree by replacing children or constructing new ones if they didnt exist
                for child_type in replacement_node.args:
                    # Retrieve the child to copy, or return None if there wasnt one
                    child_idx = next(iter(node_to_replace_children[child_type]), None)

                    # If we found the child in original tree
                    if child_idx:
                        # Then use the original child in the new replacement tree
                        slice_ = subtree_to_replace.searchSubtree(child_idx)
                        child_subtree = subtree_to_replace[slice_]
                        replacement_tree.extend(child_subtree)

                        # We've now used this child so we shouldnt be able to select it again
                        node_to_replace_children[child_type].remove(child_idx)
                    else:
                        # Otherwise make a new child
                        new_subtree = expr(pset=pset, type_=child_type)
                        replacement_tree.extend(new_subtree)

                # Place the generated subtree into the original individual
                slice_ = individual.searchSubtree(index)
                ind = toolbox.clone(individual)
                ind[slice_] = replacement_tree

                # If the resulting tree is different to original then we are done
                if str(ind) != str(individual):
                    return ind

    return toolbox.clone(individual)


def mutInsert(individual, pset, expr, toolbox):

    iprims = [(idx, node) for idx, node in enumerate(individual)
                  if isinstance(node, gp.Primitive)]

    # Search for nodes that can be extended/grown
    for i, node in shuffled(iprims):
            return_type = node.ret

            # A growable node is defined as a node with return type T for which another
            # node exists that has T as both an input and as an output
            replacement_node_types = [prim for prim in pset.primitives[return_type] if return_type in prim.args]

            original_subtree_slice_ = individual.searchSubtree(i)
            original_subtree = individual[original_subtree_slice_]

            # If this is a growable node, try and grow it
            for replacement_node_type in shuffled(replacement_node_types):

                # Make a new tree which returns the type of the replacement node
                replacement_tree = [replacement_node_type]

                # A flag to know if we have added the original subtree to our replacement tree yet
                inserted_original_subtree = False

                # Fill the children of this new replacement node
                for child_type in replacement_node_type.args:
                    # This is when we insert the original subtree
                    if not inserted_original_subtree and child_type == return_type:
                        replacement_tree.extend(original_subtree)
                        inserted_original_subtree = True
                    else:
                        # Generate the missing children
                        new_subtree = expr(pset=pset, type_=child_type)
                        replacement_tree.extend(new_subtree)

                # Place the generated subtree into the original individual
                ind = toolbox.clone(individual)
                ind[original_subtree_slice_] = replacement_tree

                # If the resulting tree was unique then we are done
                if str(ind) != str(individual):
                    return ind

    return toolbox.clone(individual)


def mutBayesian(individual, pset, expr, toolbox):
    """
    Extension of gp.mutUniform which tries to mutate individual
    to one which hasnt existed before

    :param individual: The tree to be mutated.
    :param expr: A function object that can generate an expression when
                 called.
    :returns: A tuple of one tree.
    """

    ind = bayesian_parameter_optimisation(individual, toolbox, evals=20)

    if str(ind) != str(individual):
        # Found a unique one, so return it
        return ind

    # Couldn't make a unique individual, so return None
    return toolbox.clone(individual)


def mutUniform(individual, pset, expr, toolbox, max_tries=10):
    """
    Extension of gp.mutUniform which tries to mutate individual
    to one which hasnt existed before

    :param individual: The tree to be mutated.
    :param expr: A function object that can generate an expression when
                 called.
    :returns: A tuple of one tree.
    """
    indices = list(range(len(individual)))

    # For each possible node in the tree
    for index in shuffled(indices):
        slice_ = individual.searchSubtree(index)
        type_ = individual[index].ret

        # Try several times to generate a new unique branch
        for i in range(max_tries):
            ind = toolbox.clone(individual)
            ind[slice_] = expr(pset=pset, type_=type_)

            if str(ind) != str(individual):
                # Found a unique one, so return it
                return ind

    # Couldn't make a unique individual, so return original
    return toolbox.clone(individual)


def shuffled(l):
    random.shuffle(l)
    return l
