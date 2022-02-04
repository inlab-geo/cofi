from cofi.cofi_objective.examples import XRayTomographyForward, XRayTomographyObjective
from cofi.linear_reg import LRNormalEquation

import numpy as np
import matplotlib.pyplot as plt
import os
import pytest


xrt_fwd = XRayTomographyForward()
model = np.ones([3, 3])
model[1, 1] = 2
model[0, 2] = 1.5
paths = np.array([[0, 0.5, 1, 0.9], [0, 0.5, 0.8, 0]])
print("model:\n", model)
print("paths:\n", paths)
attns = xrt_fwd.calc(model, paths)
print("attenuation:\n", attns)


def test_display(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    xrt_fwd.display(model, paths=paths, clim=(0, 2))


data_files_to_test = ["data_xrt.dat"]
@pytest.fixture(params=data_files_to_test)
def data_path():
    path_to_current_file = os.path.realpath(__file__)
    current_directory = os.path.split(path_to_current_file)[0]
    data_path = os.path.join(current_directory, "data_xrt.dat")
    return data_path

def test_xrt_obj(data_path):
    xrt_obj = XRayTomographyObjective(data_path)
    A = xrt_obj.fwd.design_matrix(xrt_obj.paths, 50, 50)
    assert A.shape[0] == xrt_obj.paths.shape[0]
    assert A.shape[1] == 50*50

def test_solving_xrt(data_path, monkeypatch):
    xrt_obj = XRayTomographyObjective(data_path)
    solver = LRNormalEquation(xrt_obj)
    model = solver.solve(0.001)
    monkeypatch.setattr(plt, "show", lambda: None)
    xrt_obj.display(model.values())

    