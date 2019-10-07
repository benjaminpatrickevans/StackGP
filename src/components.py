import src.customtypes as types
import random
from sklearn.svm import LinearSVC

def _create_estimator(method, *params):

    param_dict = {}

    for param in params:
        param_dict[param.name] = param.val

    # Required for linear SVC
    if method == LinearSVC:
        param_dict["dual"] = False

    model = method(**param_dict)

    return model

def add_estimators(pset, estimator_map, estimator_type):
    """
    Add the estimators and associated hyperparameters
    defined in estimator_map.

    :param pset:
    :param estimator_map:
    :param estimator_type:
    :return:
    """


    # For sharing parameter types
    all_parameter_types = {}

    for estimator in estimator_map:
        estimator_params = estimator_map[estimator]

        inputs = []
        for param in estimator_params:
            param_name = str(param)

            if param_name in all_parameter_types:
                # Already made this type, reuse it. This is so we can cross types over
                param_type = all_parameter_types[param_name]
            else:
                # Must add the type
                value_range = estimator_params[param]

                param_type = type(param, (), {'name': param, '__init__': types.param_init, '__str__': types.param_str,
                                    '__repr__': types.param_str})

                _add_hyperparameter(pset, param_name, param_type, value_range)

                # For recreating the types
                pset.context[param] = param_type

                all_parameter_types[param_name] = param_type

            inputs.append(param_type)

        _add_estimator(pset, estimator, inputs, estimator_type)

def _add_hyperparameter(pset, name, ret_type, value_range):
    pset.addEphemeralConstant(name, lambda: ret_type(random.choice(value_range)), ret_type)


def _add_estimator(pset, est, param_inputs, estimator_type):
    est_name = est.__name__

    # For recreating
    pset.context[est_name] = est

    # Custom parameters
    if param_inputs:
        pset.addPrimitive(lambda *params: _create_estimator(est, *params), param_inputs,
                          estimator_type, name=est_name + "Terminal")

    # Default parameters
    pset.addTerminal(_create_estimator(est), estimator_type, name=est_name + "TerminalDefault")
