from hyperopt import fmin, tpe, hp, space_eval
from hyperopt.fmin import generate_trials_to_calculate
from hyperopt.pyll.base import scope, Literal, Apply
from functools import partial
from deap import gp
import numpy as np
from hyperopt.pyll.base import Literal


# The code below is a hacky fix. len(RandomForestClassifier) throws
# an exception when called, here we can catch that exception AttributeError)
# in the try and set the length to None.
def safe__init__(self, obj=None):
    try:
        o_len = len(obj)
    except (TypeError, AttributeError):
        o_len = None
    Apply.__init__(self, 'literal', [], {}, o_len, pure=True)
    self._obj = obj

#Literal.__init__ = safe__init__

def bayesian_parameter_optimisation(tree, toolbox, evals=10):
    """
    Optimises the parameters in tree with bayesian optimisation.
    Returns a copy of the tree with the updated hyperparameters.
    :param tree:
    :return:
    """
    # TODO: Remove this try, only for debugging
    try:
        hyperparameters, hyperparameter_indices, default_values = _get_hyperparameters_from_tree(tree)
    except AttributeError as e:
        print("ERROR, the tree causing the error was:", tree)
        raise e

    if not hyperparameters:
        # Cant optimise a tree with no tunable args, so just return a copy of the original tree
        return toolbox.clone(tree)

    # Start the search at the existing values rather than randomly
    trials = generate_trials_to_calculate([default_values])

    best = fmin(
        fn=partial(_objective_function, tree, toolbox, hyperparameter_indices),
        space=hyperparameters,
        algo=tpe.suggest,
        max_evals=evals,
        trials=trials,
    )

    optimised_params = space_eval(hyperparameters, best)

    tree = _fill_with_hyperparameters(tree, toolbox, hyperparameter_indices, optimised_params)
    return tree


def _fill_with_hyperparameters(tree, toolbox, hyperparameter_indices, params):
    """
    Replaces the parameters in tree at hyperparameter_indices with those
    specified at params. Returns a copy of the tree with the replacements
    made.

    :param tree:
    :param toolbox:
    :param hyperparameter_indices:
    :param params:
    :return:
    """
    tree = toolbox.clone(tree)

    for i, param_idx in enumerate(hyperparameter_indices):
        existing_node = tree[param_idx]

        value = params[i]

        new_value = existing_node.ret(value)  # Need to wrap it in the original type for STGP

        # Replace the value with the updated one
        existing_node.value = new_value

    return tree


def _objective_function(tree, toolbox, hyperparameter_indices, *params):
    """
    What we would like to optimise. In this case we want to maximise the f1 score
    :return:

    """
    tree = _fill_with_hyperparameters(tree, toolbox, hyperparameter_indices, *params)

    # Dont save in cache or we will get lots of intermediary models in the cache
    f1 = toolbox.evaluate(tree, timeout=False, save_in_cache=False)[0]

    # Hyperopt minimises, so negate the f1
    return -f1


def _create_sampler(label, node):
    """
        Returns a sampler and default value based on the values specified in node.
        String -> hp.choice,
        Int -> hp.quniform,
        float -> hp.loguniform
    :param label:
    :param node:
    :return:
    """
    # The list of allowable values
    node_values = node.range

    # The value already set by GP
    default = node.val

    types = set(type(value) for value in node_values)

    if len(types) == 1:
        value_type = next(iter(types))

        if value_type == int:
            # Sample uniformly between the integers. Smooth function.
            return scope.int(hp.quniform(label, low=min(node_values), high=max(node_values), q=1)), default
        elif value_type == float:
            # We want to sample from log distribution due to the differing of scales, otherwise we will
            # do more exploration in larger regions. Note: we take the log when specifyig
            # low and high too, because the given scale is already as logs and this will raise to e when passed in
            safe_min = min(node_values)
            if safe_min == 0:
                # Avoid taking log of 0
                safe_min = 0.00001
            return hp.loguniform(label, low=np.log(safe_min), high=np.log(max(node_values))), default

    # Default case is to just use hp_choice. At this point we assume its a categorical or mixture of value types
    # and thus not smooth or continuous.  Note: When using hp_choice we need to return the index of the default value
    # rather than the default value itself as well
    return hp.choice(label, node_values), node_values.index(default)


def _get_hyperparameters_from_tree(tree):
    '''
    Returns all the tunable hyperparameters specified in tree.
    Tunable parameters are those which have atleast one candidate value.

    :param tree:
    :return: list, list, set -> [samplers], [parameter indices in tree], {defaults for each param}
    '''
    hyperparameters = []
    hyperparameter_indices = []
    defaults = {}

    # Find the tunable hyperparameters in the tree.
    for idx, node in enumerate(tree):
        if isinstance(node, gp.Terminal) and hasattr(node.value, "hyper_parameter")\
                and node.value.name != "seed":  # Dont try optimise random seeds

            # Cant optimise if theres only one option
            if len(node.value.range) > 1:
                sampler, default = _create_sampler(str(idx), node.value)

                hyperparameters.append(sampler)
                hyperparameter_indices.append(idx)
                defaults[str(idx)] = default

    return hyperparameters, hyperparameter_indices, defaults