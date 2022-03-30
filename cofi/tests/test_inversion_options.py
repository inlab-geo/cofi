import pytest

from cofi import InversionOptions


def test_set_unset_solving_method(capsys):
    inv_options = InversionOptions()
    inv_options.suggest_tools()
    console_output = capsys.readouterr()
    assert "optimisation" in console_output.out
    assert "numpy.linalg.lstsq" in console_output.out
    with pytest.raises(ValueError):
        inv_options.set_solving_method("abc")
    inv_options.set_solving_method("least square")
    inv_options.suggest_tools()
    console_output = capsys.readouterr()
    assert "optimisation" not in console_output.out
    assert "numpy.linalg.lstsq" in console_output.out
    inv_options.unset_solving_method()
    inv_options.suggest_tools()
    console_output = capsys.readouterr()
    assert "optimisation" in console_output.out
    assert "numpy.linalg.lstsq" in console_output.out
    inv_options.set_solving_method("least square")
    inv_options.set_solving_method(None)
    inv_options.suggest_tools()
    console_output = capsys.readouterr()
    assert "optimisation" in console_output.out
    assert "numpy.linalg.lstsq" in console_output.out

def test_set_unset_tool():
    inv_options = InversionOptions()
    with pytest.raises(ValueError):
        inv_options.set_tool("abc")
