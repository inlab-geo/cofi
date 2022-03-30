import warnings
from typing import Union, Type
import json

from .solvers import solver_suggest_table, solver_dispatch_table, solver_methods


class InversionOptions:
    def __init__(self):
        pass

    def set_solving_method(self, method: str):
        if method is None:
            self.unset_solving_method()
        elif method in solver_methods:
            self.method = method
        else:
            raise ValueError(f"the solver method is invalid, please choose from {solver_methods}")

    def unset_solving_method(self):
        del self.method

    def set_tool(self, tool: Union[str, Type]):
        if tool is None:
            self.unset_tool()
        else:
            if tool not in solver_dispatch_table:
                raise ValueError(
                    "the tool is invalid, please use `InversionOptions.suggest_tools()` to see options"
                )
            elif hasattr(self, "method") and tool not in solver_suggest_table[self.method]:
                warnings.warn(
                    f"the tool {tool} is valid but doesn't match the solving method you've selected: {self.method}"
                )
            self.tool = tool

    def unset_tool(self):
        del self.tool

    def suggest_tools(self):
        # TODO - suggest backend tool given chosen method
        if hasattr(self, "method"):
            tools = solver_suggest_table[self.method]
            print("Based on the solving method you've set, the following tools are suggested:")
            print(tools)
            print("\nUse `InversionOptions.set_tool(tool_name)` to set a specific tool from above")
            print("Use `InversionOptions.set_solving_method(tool_name)` to change solving method")
            print("Use `InversionOptions.unset_solving_method()` if you'd like to see more options")
        else:
            print("Here's a complete list of inversion solvers supported by CoFI (grouped by methods):")
            print(json.dumps(solver_suggest_table, indent=4))

    def summary(self):
        # inspiration from keras: https://keras.io/examples/vision/mnist_convnet/
        # TODO
        title = "Summary for inversion options"
        display_width = len(title)
        double_line = "=" * display_width
        single_line = "-" * display_width
        print(title)
        print(double_line)
        raise NotImplementedError       

    def __repr__(self) -> str:
        # TODO
        raise NotImplementedError
