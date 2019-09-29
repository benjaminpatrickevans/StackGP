import numpy as np


def to_numeric(data, col):
    values = list(set(data[:, col]))
    mapping = dict(zip(values, range(len(values))))
    data[:, col] = np.asarray([mapping[x] for x in data[:, col]])


def convert_to_numeric(data):
    converted_data = data[:]
    categorical_features = []
    num_features = data.shape[1]

    for i in range(num_features):
        column = converted_data[:, i]
        feature_type = type(column[0])
        num_values = len(set(column))

        # We want to convert categorical to numerical
        if feature_type == str:
            to_numeric(converted_data, i)

            if num_values > 2:
                categorical_features.append(i)

    return converted_data, categorical_features