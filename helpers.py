import pandas as pd
import numpy as np
import pickle
from sklearn import metrics
from src import sklearn_additions
from sklearn.model_selection import KFold
import time
import sys

num_folds = 10

def evaluation_score(real_y, predicted_y):
    return metrics.f1_score(real_y, predicted_y, average="weighted")


def run_and_time_classifier(model, train_x, train_y, test_x, test_y):
    start_time = time.time()
    model.fit(train_x, train_y)
    training_time_seconds = time.time() - start_time
    training_time_minutes = int(training_time_seconds / 60)

    predictions = model.predict(test_x)
    score = evaluation_score(test_y, predictions)

    return model, training_time_minutes, score


def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def _categorical_features(data_x):
    '''
        Returns the categorical column names
        from data_x. Returns an empty set
        if they are all numeric.
    :param data_x:
    :return:
    '''

    categorical_features = []

    for col in data_x:
        values = np.unique(data_x[col].values)

        # If they are all numeric
        numeric = np.all([is_number(x) for x in values])

        if not numeric:
            categorical_features.append(col)

    return categorical_features


def _fill_missing(x):
    imputer = sklearn_additions.DataFrameImputer()
    return imputer.fit_transform(x)


def read_data(data_path, class_index="last"):
    # Load the data
    data = pd.read_csv(data_path, header=None)
    data = data.replace('?', np.NaN)  # We interpret question marks as missing values
    data = data.values

    # Most datasets have class as either the first or last column
    if class_index == "last":
        data_x = data[:, :-1]
        data_y = data[:, -1]
    else:
        data_x = data[:, 1:]
        data_y = data[:, 0]

    data_x = pd.DataFrame(data_x)
    data_y = pd.DataFrame(data_y, dtype=str)

    # First thing we need to do is deal with missing values
    data_x = _fill_missing(data_x)

    # Categorical features are those where all the values are not numeric
    categorical_features = _categorical_features(data_x)

    # Convert any categorical values to numeric
    data_x = pd.get_dummies(data_x, columns=categorical_features)

    return pd.DataFrame(data_x, dtype=float), data_y


def write_file(file, contents):
    with open(file, 'wb') as handle:
        pickle.dump(contents, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run_kfold(k, dataX, dataY, fn_to_run, n_splits=10):
    print("Fold:", k)
    print("-" * 10)

    # 10-fold. Shuffle the data according to the fixed seed for reproducability
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    # We only care about the Kth fold - we run each fold as different process for speed of results
    train, test = list(kf.split(dataX, dataY))[k]

    train_x, train_y = dataX.iloc[train], dataY.iloc[train]
    test_x, test_y = dataX.iloc[test], dataY.iloc[test]

    model, training_time_minutes, score = fn_to_run(train_x, train_y, test_x, test_y)

    return score

def run(dataset, class_index, k, fn_to_run):
    data_x, dataY = read_data(dataset, class_index)
    print("Data Read")
    return run_kfold(k, data_x, dataY, fn_to_run)


def main(fn_to_run, data_folder="datasets/uci/"):
    if len(sys.argv) != 4:
        print("Must run with args: str, [first, last], int")
        print("Corresponding to: dataset_name, class_index, K")
        exit()

    dataset = data_folder + sys.argv[1] + ".data"
    class_index = sys.argv[2]
    k = int(sys.argv[3])

    print("Dataset", sys.argv[1])
    return run(dataset, class_index, k, fn_to_run)
