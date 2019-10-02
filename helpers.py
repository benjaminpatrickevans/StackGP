import pandas as pd
import numpy as np
import pickle
from sklearn import metrics
from src import sklearn_additions
from sklearn.model_selection import KFold
import time
import sys
from sklearn.preprocessing import LabelEncoder

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



def read_data(data_path):
    # Load the data
    data = pd.read_csv(data_path)
    data = data.replace('?', np.NaN)  # We interpret question marks as missing values

    x = data.drop("class", axis=1).values
    y = 'class' + data["class"].astype(str)  # In case the class is just say "1", as h2o will try do regression

    y = np.reshape(y.values, (-1, 1))  # Flatten the y so its shape (len, 1)

    return pd.DataFrame(x), pd.DataFrame(y)


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

    return training_time_minutes, score

def run(dataset, k, fn_to_run):
    data_x, data_y = read_data(dataset)
    print("Data Read")
    return run_kfold(k, data_x, data_y, fn_to_run)


def main(fn_to_run, data_folder="datasets/"):
    if len(sys.argv) != 3:
        print("Must run with args: dataset_name (str), fold (int)")
        exit()

    dataset = data_folder + sys.argv[1] + ".csv"
    k = int(sys.argv[2])

    print("Dataset", sys.argv[1])
    return run(dataset, k, fn_to_run)
