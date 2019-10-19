import signal
from contextlib import contextmanager
from sklearn.model_selection import cross_val_score, StratifiedKFold


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Cross validation timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def timed_cross_validation(model, x, y, scoring_fn, max_eval_time_seconds, n_jobs=1, num_folds=3, seed=0):
    """
    Calls sklearn.model_selection.cross_val_score but throws a
    TimeOutException if the call lasts longer than max_time
    :param model:
    :param x:
    :param y:
    :param scoring_fn:
    :param n_jobs:
    :param num_folds:
    :return:
    """
    with time_limit(max_eval_time_seconds):
        cv = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        return cross_val_score(model, x, y, cv=cv, scoring=scoring_fn, n_jobs=n_jobs).mean()