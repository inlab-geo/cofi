from .. import BaseSolver
from cofi.cofi_objective import BaseObjective
from .lib import rjmcmc


class ReversibleJumpMCMC(BaseSolver):
    def __init__(
        self,
        model=None,
        forward=None,
        x=None,
        y=None,
        error=None,
        lambda_min=None,
        lambda_max=None,
        lambda_std=None,
    ):
        """
        error: a error value per data point and can be thought as a weighting
           as to how well the fit will attempt to fit anindividual point.
           If the value is low, then the fit will be tight and
           conversely if the value is high then the fit will be loose.

        lambda: estimated noise / real noise, in cases of unknown noise std

        Note:
        x, y, and error must be of the same length
        """
        if forward:
            self.set_objective(forward)
        else:
            self.set_data(x, y, error, lambda_min, lambda_max, lambda_std)

    def set_objective(self, objective: BaseObjective):
        if objective.distance_name != "l2":
            raise ValueError("rj-MCMC package only supports l2 distance")

        self.objective = objective
        # TODO - more validation here

    def set_data(self, x, y, error, lambda_min=None, lambda_max=None, lambda_std=None):
        if x is None:
            raise ValueError("data x expected, but found None")
        if y is None:
            raise ValueError("data y expected, but found None")
        if error is None:
            raise ValueError(
                "estimated error standard deviation of data expected, but found None on"
                " parameter 'error'"
            )

        # TODO - type validation / conversion (e.g. for numpy ndarray)

        self.x = x
        self.y = y
        self.error = error

        # lambda is defined as the ratio of the estimated noise to the real noise
        # this is set to perform hierarchical Bayesian sampling
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.lambda_std = lambda_std

        if not (len(self.x) == len(self.y) and len(self.x) == len(self.error)):
            raise ValueError("Input lists must be of same length")

        self.data = rjmcmc.dataset1d(self.x, self.y, self.error)

        if lambda_min:
            self.data.set_lambda_range(self.lambda_min, self.lambda_max)
            self.data.set_lambda_std(self.lambda_std)

    def solve(
        self,
        burnin=10000,
        total=50000,
        max_order=5,
        sampler=None,
        multi_partition=False,
        partition_move_std=None,
        max_partitions=20,
        xsamples=100,
        ysamples=100,
        credible_interval=0.95,
    ):
        """
        On completion of solving, self.results is assigned a resultset1d object
        which contains various results and diagnostics about the analysis
        (Check ./lib/rjmcmc.py for inferface details)

        Parameters
        ----------
            burnin: Number
                the number of initial samples to throw away (default to 10000)
            total: Number
                the total number of samples to use for the analysis (default to 50000)
            max_order: Number
                the maximum order of polynomial to use to fit the data (default to 5)
            xsamples: Number
                the number of points to sample along the x direction for the curve (default to 100)
            ysamples: Number
                the number of points to sample along the y directory for the
                statistics such as mode, median and confidence intervals. This is
                the number of bins for the histograms in the y direction (default to 100)
            credible_interval: Number
                the confidence interval to use for minimum and maximum confidence intervals.
                This should be a value between 0 and 1 (default to 0.95)
            multi_partition: Boolean
                set to True when the data have discontinuities
            partition_move_std: Number
                the standard deviation for the perturbation of partition boundaries
                (must be set when doing multiple partitions)

        """

        if credible_interval > 1 or credible_interval < 0:
            raise ValueError("The credible interval should be between 0 and 1")

        if multi_partition:
            if partition_move_std is None:
                raise ValueError(
                    "partition_move_std required for multiple partition regression"
                )
            if sampler:
                self.results = rjmcmc.regression_part1d_sampled(
                    self.data,
                    self.gen_sampler_callback(sampler),
                    partition_move_std,
                    burnin,
                    total,
                    max_partitions,
                    max_order,
                    xsamples,
                    ysamples,
                    credible_interval,
                )
            else:
                self.results = rjmcmc.regression_part1d(
                    self.data,
                    partition_move_std,
                    burnin,
                    total,
                    max_partitions,
                    max_order,
                    xsamples,
                    ysamples,
                    credible_interval,
                )

        else:  # single partition (without discountinuities)
            if sampler:
                self.results = rjmcmc.regression_single1d_sampled(
                    self.data,
                    self.gen_sampler_callback(sampler),
                    burnin,
                    total,
                    max_order,
                    xsamples,
                    ysamples,
                    credible_interval,
                )
            else:  # solve without user's own sampling method (by default)
                self.results = rjmcmc.regression_single1d(
                    self.data,
                    burnin,
                    total,
                    max_order,
                    xsamples,
                    ysamples,
                    credible_interval,
                )

    def gen_sampler_callback(self, sampler):
        # user's own sampler takes 3 input arguments: x, y and i and returns a boolean
        # sample results are recorded in self.sample_x and self.sample_curves
        self.sample_x = None
        self.sample_curves = []
        self._sample_i = 0

        def sampler_callback(x, y):
            if self.sample_x is None:
                self.sample_x = x
            if sampler(x, y, self._sample_i):
                self.sample_curves.append(y)
            self._sample_i += 1

        return sampler_callback

    def order_historgram(self):
        if self.results is None:
            self.solve()
        return self.results.order_histogram()

    # TODO - determine whether to write a framework for results
    #        (if it's common for other approaches)
    #        or to write wrappter functions for all the functions of resultset1d
    #        (like the above order_histogram)
    #        functions include: proposed, acceptance, partitions, order_histogram,
    #                           partition_histogram, partition_location_histogram,
    #                           x, y, mean, median, mode, credible_max, misfit,
    #                           lambda_history, histogram
