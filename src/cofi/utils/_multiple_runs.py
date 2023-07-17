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
    """Run inversions on a list of problems, returning a list of InversionResult objects and a list of callback results.
    
    The inversions can be performed either sequentially or in parallel, based on the value of the 'parallel' argument.
    
    If a callback function is provided, it will be invoked for each problem with the InversionResult object and the index of the problem as arguments.
    
    Parameters
    ----------
    problems : List[BaseProblem]
        A list of problems for which the inversions need to be performed. Each problem should be an instance of the BaseProblem class.
        
    inv_options : InversionOptions
        An instance of the InversionOptions class that specifies the options to be used for the inversions.
        
    callback : Callable[[InversionResult, int], None], optional
        A callback function that gets invoked for each problem after its inversion has been performed. It should accept two arguments - the result of the inversion (an instance of InversionResult) and the index of the problem. The callback can return a result that will be collected and returned in the callback results list. If no callback function is provided, None will be used.
        
    parallel : bool, optional
        A flag that determines whether the inversions should be performed in parallel or sequentially. If True, the inversions will be performed in parallel using multiple processes. If False or not provided, the inversions will be performed sequentially in the same process.

    Returns
    -------
    Tuple[List[InversionResult], List]
        A tuple containing two lists - the first list contains the results of the inversions (as InversionResult objects) for each problem in the same order as the input problems list. The second list contains the results returned by the callback function for each problem in the same order as the input problems list.
    
    Examples
    --------

    An example of using this function to make an L-curve:
    
    >>> import cofi
    >>> import matplotlib.pyplot as plt
    >>> def my_objective(m, lamda):
    ...     data_misfit = numpy.sum((m * data_x**2 - data_y) ** 2)
    ...     regularization = lamda * numpy.sum(m**2)
    ...     return data_misfit + regularization
    ...
    >>> my_problems = []
    >>> my_lamdas = numpy.logspace(-8, 8, 30)
    >>> for lamda in my_lamdas:
    ...     my_problem = cofi.BaseProblem()
    ...     my_problem.set_objective(my_objective, args=(lamda,))
    ...     my_problem.set_initial_model(100)
    ...     my_problems.append(my_problem)
    ...
    >>> my_options = cofi.InversionOptions()
    >>> my_options.set_tool("scipy.optimize.minimize")
    >>> def my_callback(inv_result, i):
    ...     m = inv_result.model
    ...     data_misfit = numpy.linalg.norm(m * data_x**2 - data_y)
    ...     model_norm = numpy.linalg.norm(m)
    ...     return data_misfit, model_norm
    ...
    >>> all_results, all_cb_returns = cofi.utils.run_multiple_inversions(my_problems, my_options, my_callback, False)
    >>> l_curve_points = list(zip(*all_cb_returns))
    >>> plt.plot(l_curve_points)
    """
    if not problems:
        raise ValueError(
            "empty list detected, please pass in a concrete list of BaseProblem "
            "instances"
        )
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
        res, callback_res = _run_one_inversion_with_callback(problem, inv_options, i, callback)
        results.append((res, callback_res))
    results, callback_results = zip(*results)
    return results, callback_results


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
        results = list(pool.imap(_run_one_inversion_with_callback, all_args))
    results, callback_results = zip(*results)
    return results, callback_results


def _run_one_inversion_with_callback(problem, inv_options=None, i=None, callback=None):
    # for parallel case where only one argument is passed in,
    # unpack the first argument
    if not isinstance(problem, BaseProblem):
        problem, inv_options, i, callback = problem
    inv = Inversion(problem, inv_options)
    result = inv.run()
    callback_result = None
    if callback:
        callback_result = callback(result, i)
    return result, callback_result
