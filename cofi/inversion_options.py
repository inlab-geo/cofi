
class InversionOptions:
    def __init__(self):
        pass

    def setMethod(self, method: str):
        # TODO - check this
        self.method = method

    def setTool(self, tool: str):
        # TODO - check this
        self.tool = tool

    def suggets_tool(self):
        # TODO - suggest backend tool given chosen method
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError