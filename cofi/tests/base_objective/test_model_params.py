from cofi import Model

import yaml
import os
import pytest


path_to_current_file = os.path.realpath(__file__)
current_directory = os.path.split(path_to_current_file)[0]


def load_model_from_yaml(yaml_path):
    yaml_path = os.path.join(current_directory, yaml_path)
    with open(yaml_path) as f:
        cofi_spec = None
        cofi_spec = yaml.load(f, Loader=yaml.FullLoader)
        return cofi_spec["model"]


@pytest.fixture
def valid_yaml():
    return load_model_from_yaml("test_params_valid.yml")


def test_valid(valid_yaml):
    model = valid_yaml
    m = Model.init_from_yaml(model)
    m.to_yamlizable()  # also tested asdisct() in Parameter
    str(m.params)  # this is to test __repr__() in Parameter


@pytest.fixture
def invalid_yaml():
    return load_model_from_yaml("test_params_invalid.yml")


def test_invalid_yaml(invalid_yaml):
    model = invalid_yaml
    with pytest.raises(ValueError, match=r".*\*must\* contain 'parameters'.*"):
        Model.init_from_yaml(model)
    invalid_cases = [
        (1, r".*must specify 'parameters' for your model as a list.*"),
        ([1], r".*must be \(key, value\) pairs.*"),
        ([{"namee": 1}], r".*must have a 'name'.*"),
        ([{"name": "x"}], r".*no initial value AND no distribution.*"),
        ([{"name": "x", "bounds": "normm 1 2"}], r".*Unknown distribution.*"),
        ([{"name": "x", "bounds": "poisson 1"}], r".*not a continuous.*"),
        ([{"name": "x", "bounds": "poisson 1", "value": 0}], r".*not a continuous.*"),
        ([{"name": "x", "bounds": "uniform 1 2", "value": 0}], r".*zero density.*"),
        ([{"name": "x", "bounds": ["uniform 1 2"], "value": [0]}], r".*zero density.*"),
        (
            [{"name": "x", "bounds": "uniform 1 2", "value": [0, 1]}],
            r".*must be an array.*",
        ),
        (
            [{"name": "x", "bounds": ["uniform 1 2"], "value": [0, 1]}],
            r".*with same shape.*",
        ),
        (
            [{"name": "x", "bounds": ["poisson 1"], "value": [0]}],
            r".*not a continuous.*",
        ),
    ]
    for invalid_params, pattern in invalid_cases:
        model["parameters"] = invalid_params
        with pytest.raises(ValueError, match=pattern):
            Model.init_from_yaml(model)
