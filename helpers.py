
import pandas as pd
import numpy as np
import pickle
from timeit import default_timer as timer
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder


def read_data(data_path, class_index="last"):
    # Load the data
    data = pd.read_csv(data_path, header=None)
    data = data.replace('?', np.NaN)  # We interpret question marks as missing values
    data = data.values

    # Most datasets have class as either the first or last column
    if class_index == "last":
        dataX = data[:, :-1]
        dataY = data[:, -1].astype(str)  # Class label is a string
    else:
        dataX = data[:, 1:]
        dataY = data[:, 0].astype(str)  # Class label is a string


    return pd.DataFrame(dataX, dtype=float), pd.DataFrame(dataY)


def write_file(file, contents):
    with open(file, 'wb') as handle:
        pickle.dump(contents, handle, protocol=pickle.HIGHEST_PROTOCOL)


def evaluate(model, trainX, trainY, testX, testY):
    start_time = timer()

    model.fit(trainX, trainY)

    training_time = (timer() - start_time) * 1000 # Convert to milliseconds

    predictions = model.predict(testX)
    f_measure = metrics.f1_score(testY, predictions, average='weighted')
    return f_measure, training_time
