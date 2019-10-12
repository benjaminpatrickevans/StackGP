import src.customtypes as types
import random
from sklearn.pipeline import Pipeline
from sklearn.feature_selection.base import SelectorMixin
from sklearn.base import ClassifierMixin, RegressorMixin
from math import inf


class CustomPipeline(Pipeline):
    """
    Exactly the same as sklearn pipeline except for the fact that
    repr does not trim the output. Needed to recreate pipelines
    with eval(repr(pipe)).
    """

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


def add_components(pset, components, component_type, prev_step_type):
    """
    Add the components (selectors, processors, estimators) and associated hyperparameters
    given in components.

    :param pset:
    :param components:
    :param component_type:
    :return:
    """

    for component in components:
        estimator_params = components[component]

        param_inputs = []

        for param in estimator_params:
            param_name = str(component) + "_" + str(param)

            # Must add the type
            value_range = estimator_params[param]

            param_type = type(param, (), {'name': param, '__init__': types.param_init, '__str__': types.param_str,
                                '__repr__': types.param_str})

            _add_hyperparameter(pset, param_name, param_type, value_range)

            # For recreating the types
            pset.context[param] = param_type

            param_inputs.append(param_type)

        if component_type in [ClassifierMixin, RegressorMixin]:
            _add_estimator(pset, component, param_inputs, component_type, prev_step_type)
        else:
            _add_preprocessor(pset, component, param_inputs, component_type, prev_step_type)


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


def _add_preprocessor(pset, processor, param_inputs, preprocessor_type, prev_step_type):
    processor_name = processor.__name__

    # For recreating
    pset.context[processor_name] = processor

    # Cases when there is a previous step
    if prev_step_type:
        # Custom parameters
        if param_inputs:
            pset.addPrimitive(lambda prev_step, *params: _create_preprocessor(processor, prev_step, *params),
                              [prev_step_type] + param_inputs,
                              preprocessor_type, name=processor_name + "Processor")
        else:
            # No parameters. Note that unlike with classifiers the feature selectors do not all have default values
            # which is why this "else" statement exists, whereas with classifiers we add both cases
            pset.addPrimitive(lambda prev_step: _create_preprocessor(processor, prev_step),
                              [prev_step_type],
                              preprocessor_type, name=processor_name + "Processor")

    # When its the first step in pipeline
    else:
        # Custom parameters
        if param_inputs:
            pset.addPrimitive(lambda *params: _create_preprocessor(processor, [], *params),
                              param_inputs,
                              preprocessor_type, name=processor_name + "Processor")
        else:
            # No parameters. Note that unlike with classifiers the feature selectors do not all have default values
            # which is why this "else" statement exists, whereas with classifiers we add both cases
            pset.addTerminal(_create_preprocessor(processor, []), preprocessor_type, name=processor_name + "Processor")


