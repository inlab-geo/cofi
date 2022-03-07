from cofi.cofi_objective import XRayTomographyForward, XRayTomographyObjective
from cofi.linear_reg import LRNormalEquation
from cofi.optimisers import ScipyOptimiserSolver

import numpy as np
import matplotlib.pyplot as plt
import os
import pytest


@pytest.fixture
def example_model_paths():
    model = np.ones([3, 3])
    model[1, 1] = 2
    model[0, 2] = 1.5
    paths = np.array([[0, 0.5, 1, 0.9], [0, 0.5, 0.8, 0]])
    return model, paths


def test_fwd(example_model_paths):
    xrt_fwd = XRayTomographyForward()
    model, paths = example_model_paths
    print("model:\n", model)
    print("paths:\n", paths)
    attns = xrt_fwd.calc(model, paths)
    print("attenuation:\n", attns)


def test_display(monkeypatch, example_model_paths):
    xrt_fwd = XRayTomographyForward()
    model, paths = example_model_paths
    monkeypatch.setattr(plt, "show", lambda: None)
    xrt_fwd.display(model, paths=paths, clim=(0, 2))


data_files_to_test = ["data_xrt.dat", "data_xrt_uneven.dat"]


@pytest.fixture(params=data_files_to_test)
def data_path(request):
    path_to_current_file = os.path.realpath(__file__)
    current_directory = os.path.split(path_to_current_file)[0]
    data_path = os.path.join(current_directory, request.param)
    return data_path


def test_xrt_obj(data_path):
    xrt_obj = XRayTomographyObjective(data_path)
    A = xrt_obj.fwd.basis_function(xrt_obj.paths, 50, 50)
    assert A.shape[0] == xrt_obj.paths.shape[0]
    assert A.shape[1] == 50 * 50
    xrt_obj.set_grid_dimensions(60, 60)
    A = xrt_obj.fwd.basis_function(xrt_obj.paths, 60, 60)
    assert A.shape[0] == xrt_obj.paths.shape[0]
    assert A.shape[1] == 60 * 60


def test_xrt_obj_init(data_path):
    # ####### case 1: given file path
    # test #1
    with pytest.raises(ValueError, match=r".*provide observed data.*"):
        XRayTomographyObjective()
    # test #2
    with pytest.raises(ValueError, match=r".*provide a valid file path.*"):
        XRayTomographyObjective("an_invalid_path")

    # ####### case 2: given data_src_intensity, data_rec_intensity, data_paths
    # test #3
    loaded_data = np.loadtxt(data_path)
    src_intensity = loaded_data[:, 2]
    rec_intensity = loaded_data[:, 5]
    paths = np.zeros([np.shape(loaded_data)[0], 4])
    paths[:, 0] = loaded_data[:, 0]
    paths[:, 1] = loaded_data[:, 1]
    paths[:, 2] = loaded_data[:, 3]
    paths[:, 3] = loaded_data[:, 4]
    with pytest.raises(ValueError, match=r".*provide full data.*"):
        XRayTomographyObjective(src_intensity)
    # test #4
    with pytest.raises(ValueError, match=r".*dimensions between.*don't match.*"):
        XRayTomographyObjective(src_intensity, rec_intensity[0:10], paths)
    # test #5
    with pytest.raises(ValueError, match=r".*should have exactly 4 columns"):
        XRayTomographyObjective(src_intensity, rec_intensity, paths[:, 0:3])
    # test #6
    XRayTomographyObjective(src_intensity, rec_intensity, paths)
    # test #7
    with pytest.raises(ValueError, match=r".*paths data is out of bounds.*"):
        XRayTomographyObjective(
            src_intensity, rec_intensity, paths, extent=(0, 1, 0, 0.5)
        )
    # test #8
    XRayTomographyObjective(src_intensity, rec_intensity, paths, extent=(0, 1, 0, 1))

    # ####### case 3: given data_attns, data_paths
    d = -np.log(rec_intensity) + np.log(src_intensity)
    with pytest.raises(
        ValueError, match=r".*full data.*including data_attns and data_paths.*"
    ):
        XRayTomographyObjective(data_attns=d)
    with pytest.raises(
        ValueError, match=r".*dimensions between data_attns and data_paths don't.*"
    ):
        XRayTomographyObjective(data_attns=d, data_paths=paths[0:10, :])
    with pytest.raises(ValueError, match=r".*should have exactly 4 columns"):
        XRayTomographyObjective(data_attns=d, data_paths=paths[:, 0:3])
    XRayTomographyObjective(data_attns=d, data_paths=paths)


def test_xrt_misfit(example_model_paths):
    model, paths = example_model_paths
    xrt_fwd = XRayTomographyForward()
    attenuations = xrt_fwd.calc(model, paths)
    xrt_obj = XRayTomographyObjective(
        data_attns=attenuations, data_paths=paths, extent=(0, 1, 0, 1)
    )
    xrt_obj.set_grid_dimensions(3, 3)
    with pytest.raises(ValueError, match=r".*expect one in shape \(3, 3\).*"):
        xrt_obj.misfit(model[:, :2])
    with pytest.raises(ValueError, match=r".*expect one in shape \(3, 3\).*"):
        xrt_obj.misfit(model.flatten()[:3])
    assert np.isclose(xrt_obj.misfit(model), 0)


def test_solving_xrt(data_path, monkeypatch):
    xrt_obj = XRayTomographyObjective(data_path)
    # linear system solver
    linear_solver = LRNormalEquation(xrt_obj)
    with pytest.warns(UserWarning, match=r".*using linear regression formula solver.*"):
        model = linear_solver.solve(0.001)
    monkeypatch.setattr(
        plt, "show", lambda: None
    )  # comment out this line if you want to see the plot
    xrt_obj.display(model.values())
    # scipy optimiser solver
    scipy_solver = ScipyOptimiserSolver(xrt_obj)
    scipy_model = scipy_solver.solve()
    xrt_obj.display(scipy_model)
