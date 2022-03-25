import os

import pytest

from cofi import BaseProblem


data_files_to_test = [
    "datasets/dummy_test1_comma.txt",
    "datasets/dummy_test2_tab.txt",
]

@pytest.fixture(params=data_files_to_test)
def data_path(request):
    path_to_current_file = os.path.realpath(__file__)
    current_directory = os.path.split(path_to_current_file)[0]
    data_path = os.path.join(current_directory, request.param)
    return data_path

def test_set_dataset_from_file(data_path):
    inv_problem = BaseProblem()
    inv_problem.set_dataset_from_file(data_path)

def test_check_defined():
    inv_problem = BaseProblem()
    inv_problem.set_objective(lambda a: a+1)
    assert inv_problem.objective_defined

