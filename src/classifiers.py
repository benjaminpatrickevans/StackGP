from sklearn.svm import LinearSVC


def base_simple(method, *params):

    param_dict = {}

    for param in params:
        param_dict[param.name] = param.val

    # Required for linear SVC
    if method == LinearSVC:
        param_dict["dual"] = False

    model = method(**param_dict)

    return [("clf", model)]


def base(prev_steps, method, *params):

    if prev_steps is None:
        prev_steps = []

    param_dict = {}

    for param in params:
        param_dict[param.name] = param.val

    # Required for linear SVC
    if method == LinearSVC:
        param_dict["dual"] = False

    model = method(**param_dict)

    return prev_steps + [("clf", model)]