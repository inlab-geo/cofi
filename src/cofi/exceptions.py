from typing import Any, List, Union


GITHUB_ISSUE = "https://github.com/inlab-geo/cofi/issues"

class CofiError(Exception):
    """Base class for all CoFI errors"""
    pass

class InvalidOptionError(CofiError, ValueError):
    r"""Raised when user passes an invalid option into our methods / functions

    This is a subclass of :exc:`CofiError` and :exc:`ValueError`.

    """
    def __init__(self, *args, name: str, invalid_option: Any, valid_options: Union[List, str]):
        super().__init__(*args)
        self._name = name
        self._invalid_option = invalid_option
        self._valid_options = valid_options

    def __str__(self) -> str:
        super_msg = super().__str__()
        msg = f"the {self._name} you've entered ('{self._invalid_option}') is " \
              f"invalid, please choose from the following: {self._valid_options}.\n\n" \
              f"If you find it valuable to have '{self._invalid_option}' in CoFI, "\
              f"please create an issue here: {GITHUB_ISSUE}"
        if len(super_msg)>0:
            return msg+"\n\n"+super_msg
        else:
            return msg
