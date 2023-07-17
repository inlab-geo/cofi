from typing import Callable, List
import multiprocessing
import itertools

from .._base_problem import BaseProblem
from .._inversion_options import InversionOptions
from .._inversion import Inversion, InversionResult


def run_multiple_inversions(
    problems: List[BaseProblem],
    inv_options: InversionOptions,
    callback: Callable[[InversionResult, int], None] = None,
    parallel: bool = False,
) -> List[InversionResult]:
    """Run the inversion for all the problems and return the results.

    If a callback function is provided, it will be called with each InversionResult and
    index i.
    """
    if parallel:
        return _run_multiple_inversions_parallel(problems, inv_options, callback)
    else:
        return _run_multiple_inversions_sequential(problems, inv_options, callback)


def _run_multiple_inversions_sequential(
    problems: List[BaseProblem],
    inv_options: InversionOptions,
    callback: Callable[[InversionResult, int], None] = None,
) -> List[InversionResult]:
    results = []
    for i, problem in enumerate(problems):
        res = _run_one_inversion_with_callback(problem, inv_options, i, callback)
        results.append(res)
    return results


def _run_multiple_inversions_parallel(
    problems: List[BaseProblem],
    inv_options: InversionOptions,
    callback: Callable[[InversionResult, int], None] = None,
) -> List[InversionResult]:
    inv_options_list = itertools.repeat(inv_options)
    callback_list = itertools.repeat(callback)
    i_s = range(len(problems))
    all_args = zip(problems, inv_options_list, i_s, callback_list)
    with multiprocessing.Pool() as pool:
        results = list(pool.imap(_run_one_inversion_with_callback_args, all_args))
    # results = [res[0] for res in results]
    return results


def _run_one_inversion_with_callback(problem, inv_options, i, callback=None):
    inv = Inversion(problem, inv_options)
    result = inv.run()
    if callback:
        callback(result, i)
    return result


# so that the function can be pickeable and passed into pool.imap
def _run_one_inversion_with_callback_args(all_args):
    problem, inv_options, i, callback = all_args
    return _run_one_inversion_with_callback(problem, inv_options, i, callback)
