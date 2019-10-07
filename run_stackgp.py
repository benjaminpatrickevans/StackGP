from src.StackGPClassifier import StackGPClassifier
from comparisons import helpers
from functools import partial
import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    args = helpers.args()

    method = StackGPClassifier(
        max_run_time_mins=args.runtime,
        n_jobs=args.cores,
        verbose=1
    )

    fn = partial(helpers.run_and_time_classifier, method)
    scores = helpers.main(args, fn)
    print("StackGP", scores)