from sklearn import metrics
from sklearn.model_selection import KFold
from src.RandomPipelines import RandomPipelines
from helpers import read_data
from scipy.stats import ttest_ind_from_stats
import time
import sys
import warnings
warnings.filterwarnings('ignore')

# The custom name of our method
pop_size = 10000
num_folds = 10
data_folder = "datasets/uci/"  # Relative directory
seed = 0

def evaluation_score(real_y, predicted_y):
    return metrics.f1_score(real_y, predicted_y, average="weighted")


def comparison(train_x, train_y, test_x, test_y, max_running_time):

    # Track the time so we can allocate the comparison methods the same time
    start_time = time.time()

    method = RandomPipelines(pop_size=pop_size, max_running_time=max_running_time,
                             random_state=seed, verbose=1)

    method.fit(train_x, train_y)
    training_time_seconds = time.time() - start_time

    training_time_minutes = int(training_time_seconds / 60)

    predictions = method.predict(test_x)
    score = evaluation_score(test_y, predictions)
    print("Proposed:", round(score, 3), "and took", training_time_minutes, "minutes")


    return [score]


def run_kfold(K, dataX, dataY, max_running_time):
    print("Fold:", K)
    print("-" * 10)

    # Shuffle the data according to the fixed seed for reproducability
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=0)

    # We only care about the Kth fold - we run each fold as different process for speed of results
    train, test = list(kf.split(dataX, dataY))[K]

    train_x, train_y = dataX.iloc[train], dataY.iloc[train]
    test_x, test_y = dataX.iloc[test], dataY.iloc[test]

    res = comparison(train_x, train_y, test_x, test_y, max_running_time)

    print(res)

    return res


def main(dataset, class_index, K, max_running_time):
    file_name = dataset + ".data"  # Data name with extension added
    print("Dataset:", dataset)
    data_x, dataY = read_data(data_folder+file_name, class_index)
    print("Data Read")
    run_kfold(K, data_x, dataY, max_running_time)


if __name__ == "__main__":

    if len(sys.argv) != 5:
        print("Must run with args: str, [first, last], int, int")
        print("Corresponding to: dataset_name, class_index, K, training time (mins)")
        exit()


    dataset = sys.argv[1]
    class_index = sys.argv[2]
    K = int(sys.argv[3])
    max_running_time = int(sys.argv[4])
    print("Params read")

    main(dataset, class_index, K, max_running_time)
