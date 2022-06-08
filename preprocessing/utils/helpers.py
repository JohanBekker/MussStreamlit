import inspect
import gzip
import time
from contextlib import contextmanager

from pathlib import Path


def add_dicts(*dicts):
    return {k: v for dic in dicts for k, v in dic.items()}


def get_default_args(func):
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


# def failsafe_division(a, b, default=0):
#     if b == 0:
#         return default
#     return a / b


def yield_lines(filepath, gzipped=False, n_lines=None):
    filepath = Path(filepath)
    open_function = open
    if gzipped or filepath.name.endswith('.gz'):
        open_function = gzip.open
    with open_function(filepath, 'rt', encoding='utf-8') as f:
        for i, l in enumerate(f):
            if n_lines is not None and i >= n_lines:
                break
            yield l.rstrip('\n')


@contextmanager
def log_action(action_description):
    start_time = time.time()
    print(f'{action_description}...')
    try:
        yield
    except BaseException as e:
        print(f'{action_description} failed after {time.time() - start_time:.2f}s.')
        raise e
    print(f'{action_description} completed after {time.time() - start_time:.2f}s.')
