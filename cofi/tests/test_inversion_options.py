import pytest

from cofi import InversionOptions
from cofi.solvers import BaseSolver


def test_set_unset_solving_method(capsys):
    inv_options = InversionOptions()
    # 0
    inv_options.suggest_tools()
    console_output = capsys.readouterr()
    assert "optimisation" in console_output.out
    assert "numpy.linalg.lstsq" in console_output.out
    # 1
    with pytest.raises(ValueError):
        inv_options.set_solving_method("abc")
    # 2
    inv_options.set_solving_method("least square")
    inv_options.suggest_tools()
    console_output = capsys.readouterr()
    assert "optimisation" not in console_output.out
    assert "numpy.linalg.lstsq" in console_output.out
    # 3
    inv_options.unset_solving_method()
    inv_options.suggest_tools()
    console_output = capsys.readouterr()
    assert "optimisation" in console_output.out
    assert "numpy.linalg.lstsq" in console_output.out
    # 4
    inv_options.set_solving_method("least square")
    inv_options.set_solving_method(None)
    inv_options.suggest_tools()
    console_output = capsys.readouterr()
    assert "optimisation" in console_output.out
    assert "numpy.linalg.lstsq" in console_output.out

def test_set_unset_tool():
    inv_options = InversionOptions()
    # 0 - invalid input
    with pytest.raises(ValueError):
        inv_options.set_tool("abc")
    # 1 - mismatch with solving_method
    inv_options.set_solving_method("optimisation")
    with pytest.warns(UserWarning):
        inv_options.set_tool("numpy.linalg.lstsq")
    # 2 - unset
    inv_options.unset_tool()
    with pytest.raises(AttributeError):
        inv_options.tool
    # 3 - default without solving_method
    inv_options.unset_solving_method()
    assert inv_options.get_default_tool() == "scipy.optimize.minimize"
    # 4 - default given solving_method
    inv_options.set_solving_method("optimisation")
    assert inv_options.get_default_tool() == "scipy.optimize.minimize"
    # 5 - set None
    inv_options.set_tool(inv_options.get_default_tool())
    inv_options.set_tool(None)
    with pytest.raises(AttributeError):
        inv_options.tool
    # 6 - self-defined tool
    class MyOwnSolver(BaseSolver):
        def __init__(self, inv_problem, inv_options):
            pass
        def __call__(self):
            return 1
    inv_options.set_tool(MyOwnSolver)
    # 7 - self-defined invalid tool
    class AnotherSolver(BaseSolver):
        def __init__(self, inv_problem, inv_options):
            pass
    with pytest.raises(ValueError):
        inv_options.set_tool(AnotherSolver)
