from typing import Callable, List, Union, Tuple
import multiprocessing
import itertools

from .._base_problem import BaseProblem
from .._inversion_options import InversionOptions
from .._inversion import Inversion, InversionResult


class InversionPool:
    """This class manages an ensemble of inversions and allows them to be run in parallel or sequentially.

    Parameters
    ----------
    list_of_inv_problems : Union[List[BaseProblem], BaseProblem]
        A list of problems for which the inversions need to be performed, or a single problem. Each problem should be an instance of the BaseProblem class.

    list_of_inv_options : Union[List[InversionOptions], InversionOptions]
        A list of options for the inversions, or a single set of options. Each set of options should be an instance of the InversionOptions class.

    callback : Callable[[InversionResult, int], None], optional
        A callback function that gets invoked for each problem after its inversion has been performed. It should accept two arguments - the result of the inversion (an instance of InversionResult) and the index of the problem. The callback can return a result that will be collected and returned in the callback results list. If no callback function is provided, None will be used.

    parallel : bool, optional
        A flag that determines whether the inversions should be performed in parallel or sequentially. If True, the inversions will be performed in parallel using multiple processes. If False or not provided, the inversions will be performed sequentially in the same process.

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
    >>> my_ensemble_of_inversions = cofi.utils.InversionPool(my_problems, my_options, my_callback, False)
    >>> all_results, all_cb_returns = my_ensemble_of_inversions.run()
    >>> l_curve_points = list(zip(*all_cb_returns))
    >>> plt.plot(l_curve_points)
    """

    def __init__(
        self,
        list_of_inv_problems: Union[List[BaseProblem], BaseProblem],
        list_of_inv_options: Union[List[InversionOptions], InversionOptions],
        callback: Callable[[InversionResult, int], None] = None,
        parallel: bool = False,
    ):
        if not list_of_inv_problems:
            raise ValueError(
                "empty list detected, please pass in a concrete list of BaseProblem "
                "instances"
            )
        if not list_of_inv_options:
            raise ValueError(
                "empty list detected, please pass in a concrete list of"
                " InversionOptions instances"
            )
        if (
            isinstance(list_of_inv_problems, list)
            and isinstance(list_of_inv_options, list)
            and len(list_of_inv_problems) != len(list_of_inv_options)
        ):
            raise ValueError(
                "mismatch in lengths, `list_of_inv_problems` has length "
                f"{len(list_of_inv_problems)} and `list_of_inv_options has length "
                f"{len(list_of_inv_options)}. Please make sure they are of the same "
                "length"
            )
        self.problems = (
            list_of_inv_problems
            if isinstance(list_of_inv_problems, list)
            else itertools.repeat(list_of_inv_problems)
        )
        self.options = (
            list_of_inv_options
            if isinstance(list_of_inv_options, list)
            else itertools.repeat(list_of_inv_options)
        )
        self.callback = callback
        self.parallel = parallel

    def run(self):
        """
        Runs all the inversions in the ensemble, either in parallel or sequentially.

        Returns
        -------
        Tuple[List[InversionResult], List]
            A tuple containing two lists - the first list contains the results of the inversions (as InversionResult objects) for each problem in the same order as the input problems list. The second list contains the results returned by the callback function for each problem in the same order as the input problems list.
        """
        if self.parallel:
            return self._run_multiple_inversions_parallel(
                self.problems, self.options, self.callback
            )
        else:
            return self._run_multiple_inversions_sequential(
                self.problems, self.options, self.callback
            )

    def _run_multiple_inversions_sequential(
        self,
        list_of_inv_problems: List[BaseProblem],
        list_of_inv_options: List[InversionOptions],
        callback: Callable[[InversionResult, int], None] = None,
    ) -> Tuple[List[InversionResult], List]:
        results = []
        for i, (problem, inv_options) in enumerate(
            zip(list_of_inv_problems, list_of_inv_options)
        ):
            res, callback_res = self._run_one_inversion_with_callback(
                problem, inv_options, i, callback
            )
            results.append((res, callback_res))
        results, callback_results = zip(*results)
        return results, callback_results

    def _run_multiple_inversions_parallel(
        self,
        list_of_inv_problems: List[BaseProblem],
        list_of_inv_options: List[InversionOptions],
        callback: Callable[[InversionResult, int], None] = None,
    ) -> Tuple[List[InversionResult], List]:
        callback_list = itertools.repeat(callback)
        i_s = range(len(list_of_inv_problems))
        all_args = zip(list_of_inv_problems, list_of_inv_options, i_s, callback_list)
        with multiprocessing.Pool() as pool:
            results = list(pool.imap(self._run_one_inversion_with_callback, all_args))
        results, callback_results = zip(*results)
        return results, callback_results

    @staticmethod
    def _run_one_inversion_with_callback(
        problem, inv_options=None, i=None, callback=None
    ):
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
