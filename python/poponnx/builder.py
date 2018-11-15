import poponnx


# A wrapper around the Builder cpp class, renamed BuilderCore in pybind,
# to enable more pythonic use. See builder.hpp for the class definition.
class Builder(poponnx.BuilderCore):
    def __init__(self):
        super(Builder, self).__init__()
