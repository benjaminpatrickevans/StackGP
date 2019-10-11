import src.customtypes as types
import random
from sklearn.pipeline import Pipeline
from sklearn.feature_selection.base import SelectorMixin
from math import inf

class CustomPipeline(Pipeline):

    def __repr__(self, N_CHAR_MAX=inf):
        return super().__repr__(N_CHAR_MAX=N_CHAR_MAX)

    __str__ = __repr__

def _create_estimator(method, prev_steps, *params):
    param_dict = {}

    for param in params:
        param_dict[param.name] = param.val

    model = method(**param_dict)

    return CustomPipeline(steps=prev_steps + [("clf", model)])


def _create_preprocessor(method, prev_steps, *params):

    param_dict = {}

    for param in params:
        param_dict[param.name] = param.val

    model = method(**param_dict)

    return prev_steps + [("Preprocessor" + str(len(prev_steps)), model)]


def add_estimators(pset, estimator_map, estimator_type):
    """
    Add the estimators and associated hyperparameters
    defined in estimator_map.

    :param pset:
    :param estimator_map:
    :param estimator_type:
    :return:
    """

    prev_step = SelectorMixin

    for estimator in estimator_map:
        estimator_params = estimator_map[estimator]

        param_inputs = []

        for param in estimator_params:
            param_name = str(estimator) + "_" + str(param)

            # Must add the type
            value_range = estimator_params[param]

            param_type = type(param, (), {'name': param, '__init__': types.param_init, '__str__': types.param_str,
                                '__repr__': types.param_str})

            _add_hyperparameter(pset, param_name, param_type, value_range)

            # For recreating the types
            pset.context[param] = param_type

            param_inputs.append(param_type)

        _add_estimator(pset, estimator, param_inputs, estimator_type, prev_step)


def _add_hyperparameter(pset, name, ret_type, value_range):
    pset.addEphemeralConstant(name, lambda: ret_type(random.choice(value_range)), ret_type)


def _add_estimator(pset, est, param_inputs, estimator_type, prev_step_type):
    est_name = est.__name__

    # For recreating
    pset.context[est_name] = est

    # Custom parameters for the estimator
    if param_inputs:
        pset.addPrimitive(lambda prev_step, *params: _create_estimator(est, prev_step, *params),
                          [prev_step_type] + param_inputs,
                          estimator_type, name=est_name+"Estimator")

    # Default parameters for the estimator
    pset.addPrimitive(lambda prev_step: _create_estimator(est, prev_step),
                      [prev_step_type],
                      estimator_type, name=est_name + "EstimatorDefault")


def add_feature_preprocessors(pset, preprocessor_map):
    """
    Add the estimators and associated hyperparameters
    defined in estimator_map.

    :param pset:
    :param preprocessor_map:
    :param estimator_type:
    :return:
    """

    estimator_type = SelectorMixin

    for preprocessor in preprocessor_map:
        preprocessor_params = preprocessor_map[preprocessor]

        inputs = []
        for param in preprocessor_params:
            param_name = str(preprocessor) + "_" + str(param)

            # Must add the type
            value_range = preprocessor_params[param]

            param_type = type(param, (), {'name': param, '__init__': types.param_init, '__str__': types.param_str,
                                '__repr__': types.param_str})

            _add_hyperparameter(pset, param_name, param_type, value_range)

            # For recreating the types
            pset.context[param] = param_type

            inputs.append(param_type)

        _add_feature_preprocessor(pset, preprocessor, inputs, estimator_type)


def _add_feature_preprocessor(pset, processor, param_inputs, estimator_type):
    processor_name = processor.__name__

    # For recreating
    pset.context[processor_name] = processor

    # Custom parameters
    if param_inputs:
        pset.addPrimitive(lambda *params: _create_preprocessor(processor, [], *params), param_inputs,
                          estimator_type, name=processor_name + "Selector")
    else:
        # No parameters. Note that unlike with classifiers the feature selectors do not all have default values
        # which is why this "else" statement exists, whereas with classifiers we add both cases
        pset.addTerminal(_create_preprocessor(processor, []), estimator_type, name=processor_name + "Selector")