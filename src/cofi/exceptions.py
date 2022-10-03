from typing import Any, List, Tuple, Union


GITHUB_ISSUE = "https://github.com/inlab-geo/cofi/issues"


class CofiError(Exception):
    """Base class for all CoFI errors"""

    def _form_str(self, super_msg, msg):
        return f"{msg}\n\n{super_msg}" if super_msg else msg


class InvalidOptionError(CofiError, ValueError):
    r"""Raised when user passes an invalid option into our methods / functions

    This is a subclass of :exc:`CofiError` and :exc:`ValueError`.

    Parameters
    ----------
    *args : Any
        passed on directly to :exc:`ValueError`
    name: str
        name of the item that tries to take the invalid option
    invalid_option : Any
        the invalid option entered
    valid_options : list or str
        a list of valid options to choose from, or a string describing valid options
    """

    def __init__(
        self, *args, name: str, invalid_option: Any, valid_options: Union[List, str]
    ):
        super().__init__(*args)
        self._name = name
        self._invalid_option = invalid_option
        self._valid_options = valid_options

    def __str__(self) -> str:
        super_msg = super().__str__()
        msg = (
            f"the {self._name} you've entered ('{self._invalid_option}') is "
            f"invalid, please choose from the following: {self._valid_options}.\n\n"
            f"If you find it valuable to have '{self._invalid_option}' for "
            f"{self._name} in CoFI, please create an issue here: {GITHUB_ISSUE}"
        )
        return self._form_str(super_msg, msg)


class DimensionMismatchError(CofiError, ValueError):
    r"""Raised when model or data shape doesn't match existing problem settings

    This is a subclass of :exc:`CofiError` and :exc:`ValueError`.

    Parameters
    ----------
    *args : Any
        passed on directly to :exc:`ValueError`
    entered_dimension : tuple
        dimension entered that conflicts with existing one
    entered_name : str
        name of the item, the dimension of which is entered
    expected_dimension : tuple
        dimension expected based on existing information
    expected_source : str
        name of an existing component that infers ``expected_dimension``
    """

    def __init__(
        self,
        *args,
        entered_dimension: Tuple,
        entered_name: str,
        expected_dimension: Tuple,
        expected_source: str,
    ) -> None:
        super().__init__(*args)
        self._entered_dimension = entered_dimension
        self._entered_name = entered_name
        self._expected_dimension = expected_dimension
        self._expected_source = expected_source

    def __str__(self) -> str:
        super_msg = super().__str__()
        msg = (
            f"the {self._entered_name} you've provided (shape: "
            f"{self._entered_dimension}) doesn't match and cannot be reshaped "
            f"into the dimension you've set for {self._expected_source} which is "
            f"{self._expected_dimension}"
        )
        return self._form_str(super_msg, msg)


class NotDefinedError(CofiError, NotImplementedError):
    r"""Raised when a certain property or function is not set to a :class:BaseProblem
    instance but attempts are made to use it (e.g. in a solving approach)

    This is a subclass of :exc:`CofiError` and :exc:`NotImplementedError`.

    Parameters
    ----------
    *args : Any
        passed on directly to :exc:`NotImplementedError`
    needs : list or str
        a list of information required to perform the operation, or a string describing
        them
    """

    def __init__(self, *args, needs: Union[List, str]):
        super().__init__(*args)
        self._needs = needs

    def __str__(self) -> str:
        super_msg = super().__str__()
        msg = (
            f"`{self._needs}` is required in the solving approach but you haven't "
            "implemented or added it to the problem setup"
        )
        return self._form_str(super_msg, msg)


class InvocationError(CofiError, RuntimeError):
    r"""Raised when there's an error happening during excecution of a function

    This is a subclass of :exc:`CofiError` and :exc:`RuntimeError`.

    One should raise this error by ``raise InvocationError(func_name=a, autogen=b) from exception``,
    where ``exception`` is the original exception caught

    Parameters
    ----------
    *args : Any
        passed on directly to :exc:`RuntimeError`
    func_name : str
        name of the function that runs into error
    autogen : bool
        whether this function is automatically generated or defined by users
    """

    def __init__(self, *args, func_name: str, autogen: bool):
        super().__init__(*args)
        self._func_name = func_name
        self._func_name_prefix = "auto-generated" if autogen else "your"

    def __str__(self) -> str:
        super_msg = super().__str__()
        msg = (
            f"exception while calling {self._func_name_prefix} {self._func_name}. "
            "Check exception details from message above. If not sure, "
            f"please report this issue at {GITHUB_ISSUE}"
        )
        return self._form_str(super_msg, msg)
