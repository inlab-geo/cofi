"""Introspection of CoFI's live capabilities.

Grounds external tooling (e.g. an LLM assistant scaffolding a
:class:`~cofi.BaseProblem`) in the *installed* CoFI, instead of figures
hardcoded against a particular release that go stale on the next one. Kept to
the standard library only (dataclasses, no pydantic) so it adds no dependency
to CoFI core; anything richer belongs in a downstream package that depends on
``cofi``, not the other way around.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


__all__ = [
    "ToolCapabilities",
    "CofiCapabilities",
    "describe_capabilities",
]


@dataclass(frozen=True)
class ToolCapabilities:
    r"""The requirement matrix for a single backend inference tool

    Mirrors the four classmethods every :class:`~cofi.tools.BaseInferenceTool`
    subclass defines, plus the display fields and the ``inv_options.method``
    group it's dispatched under.
    """

    #: the ``inv_options.tool`` registry key, e.g. ``"emcee"``
    name: str
    #: the ``inv_options.method`` group, e.g. ``"sampling"``
    method: str
    #: :class:`~cofi.BaseProblem` components that must be set
    required_in_problem: List[str] = field(default_factory=list)
    #: :class:`~cofi.BaseProblem` components that may optionally be set
    optional_in_problem: List[str] = field(default_factory=list)
    #: solver-specific options that must be set on :class:`~cofi.InversionOptions`
    required_in_options: List[str] = field(default_factory=list)
    #: solver-specific options that may optionally be set, mapped to their default
    optional_in_options: Dict[str, Any] = field(default_factory=dict)
    short_description: str = ""
    documentation_links: List[str] = field(default_factory=list)
    #: ``False`` when the requirement matrix couldn't be fully introspected
    #: because the backend's own optional dependency isn't installed
    available: bool = True
    unavailable_reason: Optional[str] = None


@dataclass(frozen=True)
class CofiCapabilities:
    r"""A snapshot of what the installed CoFI package currently supports

    Returned by :func:`describe_capabilities`. Everything here is derived by
    reading the installed package at call time, so it always matches the
    running ``cofi`` version.
    """

    #: the running ``cofi.__version__``
    cofi_version: str
    #: the full component vocabulary, i.e. :attr:`cofi.BaseProblem.all_components`
    components: List[str]
    #: the solving method groups, e.g. ``["matrix solvers", "optimization", "sampling"]``
    solving_methods: List[str]
    #: one entry per registered backend tool (``cofi.tools.tool_dispatch_table`` key)
    tools: List[ToolCapabilities]

    def as_dict(self) -> dict:
        r"""Convert this snapshot into a plain, JSON-serialisable ``dict``"""
        return asdict(self)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return repr(value)


def _as_str(description) -> str:
    # a couple of tool wrappers define `short_description` as a list by mistake
    if isinstance(description, str):
        return description
    return " ".join(description)


def _safe_call(fn, default):
    r"""Call a requirement-matrix classmethod, tolerating a missing optional dep

    A handful of backend tools (e.g. ``emcee``, ``bayesbay``) derive their
    options from the wrapped library's own signature, which means calling
    them imports that library. If it isn't installed, we still want the rest
    of the snapshot to build successfully.
    """
    try:
        return fn(), None
    except ImportError as exc:
        return default, str(exc)


def _tool_capabilities(name: str, method: str, tool_cls) -> ToolCapabilities:
    required_in_problem, err1 = _safe_call(tool_cls.required_in_problem, set())
    optional_in_problem, err2 = _safe_call(tool_cls.optional_in_problem, {})
    required_in_options, err3 = _safe_call(tool_cls.required_in_options, set())
    optional_in_options, err4 = _safe_call(tool_cls.optional_in_options, {})
    reason = next((e for e in (err1, err2, err3, err4) if e), None)

    return ToolCapabilities(
        name=name,
        method=method,
        required_in_problem=sorted(required_in_problem),
        optional_in_problem=sorted(optional_in_problem),
        required_in_options=sorted(required_in_options),
        optional_in_options={
            k: _json_safe(v) for k, v in sorted(optional_in_options.items())
        },
        short_description=_as_str(tool_cls.short_description),
        documentation_links=list(tool_cls.documentation_links),
        available=reason is None,
        unavailable_reason=(
            None
            if reason is None
            else f"optional dependency for '{name}' is not installed: {reason}"
        ),
    )


def describe_capabilities() -> CofiCapabilities:
    r"""Introspect the installed CoFI package for its current capabilities

    Reads the component vocabulary from :attr:`cofi.BaseProblem.all_components`
    and the requirement matrix for every registered backend tool from
    :data:`cofi.tools.inference_tools_table`, so the result always reflects
    what the *installed* ``cofi`` actually supports rather than figures frozen
    at some earlier release.

    This is the intended integration point for anything that needs to ground
    itself in CoFI's live capabilities (for example an LLM assistant
    generating a :class:`~cofi.BaseProblem` setup): call this at request time
    rather than hardcoding the component or solver list.

    Returns
    -------
    CofiCapabilities
        a snapshot of the components, solving methods, and per-tool
        requirement matrix, derived fresh from the installed package

    Examples
    --------

    .. admonition:: example usage for cofi.describe_capabilities()
        :class: dropdown, attention

        .. code-block:: pycon

            >>> import cofi
            >>> capabilities = cofi.describe_capabilities()
            >>> "objective" in capabilities.components
            True
            >>> emcee = next(t for t in capabilities.tools if t.name == "emcee")
            >>> emcee.required_in_problem
            ['log_posterior']
            >>> sorted(emcee.required_in_options)
            ['initial_state', 'nsteps', 'nwalkers']

    """
    from .._base_problem import BaseProblem
    from .._version import __version__
    from . import inference_tools_table

    tools = [
        _tool_capabilities(name, method, tool_cls)
        for method, backend_tools in inference_tools_table.items()
        for name, tool_cls in backend_tools.items()
    ]
    tools.sort(key=lambda tool: tool.name)

    return CofiCapabilities(
        cofi_version=__version__,
        components=list(BaseProblem.all_components),
        solving_methods=sorted(inference_tools_table.keys()),
        tools=tools,
    )
