import poponnx


# A wrapper around the Builder cpp class, renamed BuilderCore in pybind,
# to enable more pythonic use. See builder.hpp for the class definition.
class Builder(poponnx.BuilderCore):
    def __init__(self, modelProtoOrFilename=None):
        if modelProtoOrFilename is None:
            super(Builder, self).__init__()
        else:
            super(Builder, self).__init__(modelProtoOrFilename)
