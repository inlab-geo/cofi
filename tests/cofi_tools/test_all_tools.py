# ############################### TEST ALL TOOLS ###############################
# 1. The inference tool is added to tools/__init__.py
#        __all__, and inference_tools_table
# 2. The four class methods are implemented and they return correct data types
#        required_in_problem, optional_in_problem
#        required_in_options, optional_in_options
# 3. __call__ method returns a dictionary with at least "success" and "model"
#    as keys
# 4. available_algorithms return correct data types if implemented
# ################################################################################

import pathlib
import os
import pytest

import cofi


PKG_NAME = "cofi"
ROOT = str(pathlib.Path(__file__).resolve().parent.parent.parent)
TOOLS_FOLDER = ROOT + "/src/cofi/tools"

def all_tools_from_folder():
    file_names = []
    for name in os.listdir(TOOLS_FOLDER):
        if name.startswith("_") and name != "__init__.py" \
            and name.endswith(".py") and "base_inference" not in name:
            file_names.append(name)
    tool_names = {name[1:-3] for name in file_names}
    return tool_names

def all_tools_from_init_all():
    tool_names = {name for name in cofi.tools.__all__ if name != "BaseInferenceTool"}
    return tool_names

def all_tools_from_init_table():
    table = cofi.tools.inference_tools_table
    tool_names = {table[m][t].__name__ for m in table for t in table[m] }
    return tool_names

@pytest.fixture(params=all_tools_from_init_all())
def inference_tool_class(request):
    tool_name = request.param
    return getattr(cofi.tools, tool_name)

def test_tools_included():
    tools_from_folder = all_tools_from_folder()
    tools_from_init_all = all_tools_from_init_all()
    tools_from_init_table = all_tools_from_init_table()
    # check all_tools_from_init_all == all_tools_from_init_table
    assert len(tools_from_init_all.difference(tools_from_init_table)) == 0
    # TODO check all_tools_from_folder are included

def test_each_inference_tool(inference_tool_class):
    # TODO test __init__ and __call__
    # TODO test required class methods
    # test optional class methods
    try: algs = inference_tool_class.available_algorithms()
    except AttributeError: pass
    else: assert isinstance(algs, set)
