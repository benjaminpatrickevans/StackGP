import pandas as pd
import numpy as np
import pickle
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import time
import argparse


def clf_score(real_y, predicted_y):
    return metrics.f1_score(real_y, predicted_y, average="weighted")


def reg_score(real_y, predicted_y):
    return metrics.mean_squared_error(real_y, predicted_y, multioutput="uniform_average")


def run_and_time_estimator(model, train_x, train_y, test_x, test_y, classification=True, callback=None):
    start_time = time.time()

    if callback:
        model.fit(train_x, train_y, callback=callback)
    else:
        model.fit(train_x, train_y)

    training_time_seconds = time.time() - start_time
    training_time_minutes = int(training_time_seconds / 60)

    predictions = model.predict(test_x)

    if classification:
        score = clf_score(test_y, predictions)
    else:
        score = reg_score(test_y, predictions)

    return model, training_time_minutes, score


def read_data(data_path, classification=True):
    # Load the data
    data = pd.read_csv(data_path)
    data = data.replace('?', np.NaN)  # We interpret question marks as missing values

    x = data.drop("class", axis=1).values

    y = data["class"]
    y = np.reshape(y.values, (-1, 1))  # Flatten the y so its shape (len, 1)

    if classification:
        # Need to make outputs integers for sklearn
        le = LabelEncoder()
        y = le.fit_transform(y)

    return pd.DataFrame(x), pd.DataFrame(y)


def write_file(file, contents):
    with open(file, 'wb') as handle:
        pickle.dump(contents, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run_kfold(k, dataX, dataY, fn_to_run, seed, classification, num_folds=2):
    print("Fold:", k, "\nSeed:", seed)
    print("-" * 10)

    # 10-fold. Shuffle the data according to the fixed seed for reproducability
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

    # We only care about the Kth fold - we run each fold as different process for speed of results
    train, test = list(kf.split(dataX, dataY))[k]

    train_x, train_y = dataX.iloc[train], dataY.iloc[train]
    test_x, test_y = dataX.iloc[test], dataY.iloc[test]

    model, training_time_minutes, score = fn_to_run(train_x, train_y, test_x, test_y, classification)

    return training_time_minutes, score


def run(dataset, k, fn_to_run, seed, classification):
    data_x, data_y = read_data(dataset, classification)
    return run_kfold(k, data_x, data_y, fn_to_run, seed, classification)


def main(args, fn_to_run, classification=True):
    dataset = args.dataset
    k = args.fold
    seed = args.seed

    print("Time given:", args.runtime, "minutes")
    print("Cores:", args.cores)

    return run(dataset, k, fn_to_run, seed, classification)


def args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--runtime', help='How long (in minutes) to run for', type=int)
    parser.add_argument('--seed', help='Seed for CV', type=int)
    parser.add_argument('--fold', help='Fold', type=int)
    parser.add_argument('--dataset', help='Dataset', type=str)
    parser.add_argument('--tmp', help='Temporary out folder', type=str)
    parser.add_argument('--cores', default=1, help='Number of cores to use', type=int)

    args = parser.parse_args()
    return args
