import json

import cofi
from cofi.tools import (
    CofiCapabilities,
    ToolCapabilities,
    describe_capabilities,
    inference_tools_table,
)
from cofi._base_problem import BaseProblem


def test_top_level_reexport():
    assert cofi.describe_capabilities is describe_capabilities


def test_snapshot_shape():
    snapshot = describe_capabilities()
    assert isinstance(snapshot, CofiCapabilities)
    assert snapshot.cofi_version == cofi.__version__
    assert snapshot.components == BaseProblem.all_components
    assert set(snapshot.solving_methods) == set(inference_tools_table.keys())


def test_all_registered_tools_present():
    snapshot = describe_capabilities()
    expected_names = {
        name for tools in inference_tools_table.values() for name in tools
    }
    actual_names = {tool.name for tool in snapshot.tools}
    assert actual_names == expected_names
    for tool in snapshot.tools:
        assert isinstance(tool, ToolCapabilities)


def test_tool_method_matches_dispatch_table():
    snapshot = describe_capabilities()
    by_name = {tool.name: tool for tool in snapshot.tools}
    for method, tools in inference_tools_table.items():
        for name in tools:
            assert by_name[name].method == method


def test_emcee_requirement_matrix():
    # the concrete example from the design doc: emcee requires log_posterior
    # in the problem and nwalkers/nsteps/initial_state in the options
    snapshot = describe_capabilities()
    emcee = next(tool for tool in snapshot.tools if tool.name == "emcee")
    assert emcee.required_in_problem == ["log_posterior"]
    assert set(emcee.required_in_options) == {
        "nwalkers",
        "nsteps",
        "initial_state",
    }


def test_snapshot_is_json_serialisable():
    snapshot = describe_capabilities()
    serialised = json.dumps(snapshot.as_dict())
    assert json.loads(serialised)["cofi_version"] == snapshot.cofi_version


def test_available_tool_has_no_unavailable_reason():
    snapshot = describe_capabilities()
    for tool in snapshot.tools:
        if tool.available:
            assert tool.unavailable_reason is None
        else:
            assert tool.unavailable_reason is not None
