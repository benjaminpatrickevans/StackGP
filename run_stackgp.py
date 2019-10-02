from src.StackGP import StackGP
import helpers
from functools import partial
import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    print("Num cores", helpers.num_cores)
    method = StackGP(pop_size=100, max_run_time_mins=1, verbose=1)
    fn = partial(helpers.run_and_time_classifier, method)
    scores = helpers.main(fn)
    print("StackGP", scores)