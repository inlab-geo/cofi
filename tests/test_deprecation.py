import pytest

def test_deprecated_cofi_solvers():
    from cofi import solvers
    import cofi.solvers
    from cofi.solvers import BaseInferenceTool

def test_deprecated_base_solver():
    from cofi.solvers import BaseSolver
    with pytest.raises(TypeError):
        a = BaseSolver()
