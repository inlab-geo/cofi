from typing import Type

import emcee
import arviz

from . import BaseProblem, InversionOptions
from .solvers import solver_dispatch_table, BaseSolver


class InversionResult:
    r"""The result class of an inversion run.

    You won't need to create an object of this class by yourself. See :func:`Inversion.run`
    for how you will get such an instance.

    .. seealso::

        When using sampling methods, you get a :class:`SamplingResult` object, with 
        additional analysis methods attached.

    """

    #: bool: indicates status of the inversion run
    success: bool
    #: dict: raw output from backend solvers
    res: dict

    def __init__(self, res: dict) -> None:
        self.__dict__.update(res)
        self.res = res
        if "success" not in res:
            raise ValueError(
                "inversion termination status not returned in result dictionary, "
                "fix your solver to return properly. Check CoFI documentation "
                "'tutorial - Advanced Usage' section for how to plug in your own solver"
            )
        self.success_or_not = (
            "success" if hasattr(self, "success") and self.success else "failure"
        )

    def summary(self) -> None:
        """Helper method that prints a summary of the inversion result to console"""
        self._summary()

    def _summary(self, display_lines=True) -> None:
        title = "Summary for inversion result"
        display_width = len(title)
        double_line = "=" * display_width
        single_line = "-" * display_width
        if display_lines:
            print(double_line)
        print(title)
        if display_lines:
            print(double_line)
        print(self.success_or_not.upper())
        if display_lines:
            print(single_line)
        for key, val in self.res.items():
            if key != "success":
                print(f"{key}: {val}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.success_or_not})"


class SamplingResult(InversionResult):
    """the result class of an inversion run, when the inversion is sampling-based

    This is a subclass of :class:`InversionResult`, so has the full functionality of 
    it. Additionally, you can convert a :class:`SamplingResult` object into an 
    :class:`arviz.InferenceData` object so that various plotting functionalities are
    available from arviz.

    """
    def __init__(self, res: dict) -> None:
        super().__init__(res)
        if "sampler" not in res:
            raise ValueError(
                "sampler not found in class SamplingResult, very likely to be a bug on"
                " our (CoFI) side. Please file an issue at "
                "https://github.com/inlab-geo/cofi/issues, thanks!"
            )

    def to_arviz(self, **kwargs):
        """convert sampler result into an :class:`arviz.InferenceData` object

        Note that this method takes in keyword arguments that matches the 
        ``arviz.from_<library>`` function. If your results are sampled from emcee,
        then you can pass in any keyword arguments as described in 
        :func:`arviz.from_emcee`.

        Returns
        -------
        arviz.InferenceData
            an :class:`arviz.InferenceData` object converted from your sampler

        Raises
        ------
        NotImplementedError
            when sampling result of current type (``type(SamplingResult.sampler)``))
            cannot be converted into an :class:`arviz.InferenceData`
        """
        sampler = self.sampler
        if sampler is None:
            raise ValueError(
                "sampling result is None, please double check that your solver returns"
                " correctly if you are using your own solver; otherwise please file an"
                " issue at https://github.com/inlab-geo/cofi/issues, thanks!"
            )
        if isinstance(sampler, emcee.EnsembleSampler):
            if hasattr(self, "blob_names") and self.blob_names and "blob_names" not in kwargs:
                if "blob_groups" in kwargs:
                    self.arviz_inference_data = arviz.from_emcee(
                        sampler,
                        blob_names=self.blob_names,
                        **kwargs,
                    )
                else:
                    blobs_groups = [
                        ("prior" if name == "log_prior" else name) for name in self.blob_names
                    ]       # "log_prior" isn't in arviz's supported groups, use "prior"
                    self.arviz_inference_data = arviz.from_emcee(
                        sampler,
                        blob_names=self.blob_names,
                        blob_groups=blobs_groups,
                        **kwargs,
                    )
            else:
                self.arviz_inference_data = arviz.from_emcee(sampler, **kwargs)
            return self.arviz_inference_data
        else:
            raise NotImplementedError(
                f"sampling result of type {sampler.__class__.__name__} not supported"
                " yet, please file an issue at https://github.com/inlab-geo/cofi/issues"
                ", thanks!"
            )


class Inversion:
    r"""The class holder that take in both an inversion problem setup :class:`BaseProblem`
    and inversion solver options :class:`InversionOptions`, and handles the running of an
    inversion.

    Recall that we have 4 main steps to define and run an inversion through ``cofi``:

    1. Define a :class:`BaseProblem` object
    2. Define an :class:`InversionOptions` object
    3. Pass both of the above objects into an :class:`Inversion`
    4. Hit that :func:`Inversion.run` method and get the result :class:`InversionResult`

    So let's think of ``Inversion`` as an engine that manages the input and output of an
    inversion run for you.

    .. admonition:: Example usage of Inversion
        :class: attention

        >>> from cofi import BaseProblem, InversionOptions, Inversion
        >>> inv_problem = BaseProblem()
        >>> inv_problem.set_... # attach info about your problem
        >>> inv_options = InversionOptions()
        >>> inv_options.set_... # select backend tool and solver-specific parameters
        >>> inv = Inversion(inv_problem, inv_options)
        >>> inv_result = inv.run()

        See our example gallery for more inversion runs.

    .. admonition:: A future direction?
        :class: seealso

        We seperate out this "inversion", instead of passing a ``BaseProblem`` object
        directly to a hypothetical ``InversionSolver`` concept. This is not only because
        we want a cleaner workflow, but also because we imagine this ``Inversion``
        object to have more capability::

        >>> inv = Inversion(inv_problem, inv_options)
        >>> inv_result = inv.run()
        >>> inv.save("filename")
        >>> inv = Inversion.load("filename")
        >>> inv.analyse("filename")

    """

    def __init__(self, inv_problem: BaseProblem, inv_options: InversionOptions) -> None:
        self.inv_problem = inv_problem
        self.inv_options = inv_options
        # dispatch inversion_solver from self.inv_options, validation is done by solver
        self.inv_solve = self._dispatch_solver()(inv_problem, inv_options)
        self.inv_result = None

    def run(self) -> InversionResult:
        """Starts the inversion and returns an :class:`InversionResult` object.

        The inversion will be entirely based on the setup defined in ``BaseProblem`` and
        ``InversionOptions`` objects.

        Returns
        -------
        InversionResult
            the result of inversion that has attributes ``model`` / ``models`` and ``success``
            minimally. Check :class:`InversionResult` for details.
        """
        res_dict = self.inv_solve()
        if "sampler" in res_dict:
            self.inv_result = SamplingResult(res_dict)
        else:
            self.inv_result = InversionResult(res_dict)
        return self.inv_result

    def _dispatch_solver(self) -> Type[BaseSolver]:
        tool = self.inv_options.get_tool()
        # look up solver_dispatch_table to return constructor for a BaseSolver subclass
        if isinstance(tool, str):
            return solver_dispatch_table[tool]
        # self-defined BaseSolver (note that a BaseSolver object is a callable)
        return self.inv_options.tool

    def summary(self):
        r"""Helper method that prints a summary of current ``Inversion`` object
        to console

        This is essentially a higher level method that calls the ``.summary()`` method
        on all of the three objects:

        - :class:`InversionResult` (if the inversion has finished)
        - :class:`BaseProblem`
        - :class:`InversionOptions`

        """
        title = "Summary for Inversion"
        subtitle_result = "Completed with the following result:"
        subtitle_options = "With inversion solver defined as below:"
        subtitle_problem = "For inversion problem defined as below:"
        display_width = max(
            len(title),
            len(subtitle_result),
            len(subtitle_options),
            len(subtitle_problem),
        )
        double_line = "=" * display_width
        single_line = "-" * display_width
        print(double_line)
        print(title)
        print(double_line)
        if self.inv_result:
            print(f"{subtitle_result}\n")
            self.inv_result._summary(False)
            print(single_line)
        else:
            print("Inversion hasn't started, try `inversion.run()` to see result")
            print(single_line)
        print(f"{subtitle_options}\n")
        self.inv_options._summary(False)
        print(single_line)
        print(f"{subtitle_problem}\n")
        self.inv_problem._summary(False)
        if self.inv_result:
            print("List of functions/properties got used by the backend tool:")
            print(self.inv_solve.components_used)
