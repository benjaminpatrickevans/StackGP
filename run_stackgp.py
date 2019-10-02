from src.StackGP import StackGP
import helpers
from functools import partial
import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    method = StackGP(pop_size=100, max_run_time_mins=2, verbose=1)
    fn = partial(helpers.run_and_time_classifier, method)
    scores = helpers.main(fn)
    print("StackGP", scores)
