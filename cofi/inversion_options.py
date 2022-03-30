
from typing import Union, Type


class InversionOptions:
    def __init__(self):
        pass

    def setMethod(self, method: str):
        # TODO - check this
        self.method = method

    def setTool(self, tool: Union[str, Type]):
        # TODO - check this
        self.tool = tool

    def suggets_tools(self):
        # TODO - suggest backend tool given chosen method
        raise NotImplementedError

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