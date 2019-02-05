import poponnx


# A wrapper around the Builder cpp class, renamed BuilderCore in pybind,
# to enable more pythonic use. See builder.hpp for the class definition.
class Builder(poponnx.BuilderCore):

    #    def __init__(self, modelProtoOrFilename=None):
    #        if modelProtoOrFilename is None:
    #            super(Builder, self).__init__()
    #        else:
    #            super(Builder, self).__init__(modelProtoOrFilename)

    def __init__(self, opsets=None, modelProtoOrFilename=None):

        if (opsets == None):

            # These are the default opsets, they will increment with releases

            self.opsets = {"ai.onnx": 9, "ai.onnx.ml": 1, "ai.graphcore": 1}
        else:
            self.opsets = opsets

        if modelProtoOrFilename is None:
            super(Builder, self).__init__()
        else:
            super(Builder, self).__init__(modelProtoOrFilename)

            # After we have loaded the model we should set the opset based
            # on what is in the file.

    @property
    def aiOnnx(self):
        version = self.opsets["ai.onnx"]
        if version == 9:
            return self.aiOnnxOpset9
        elif version == 8:
            return self.aiOnnxOpset8
        elif version == 7:
            return self.aiOnnxOpset7
        elif version == 6:
            return self.aiOnnxOpset6
        else:
            # Need to raise an exception here
            pass

    @property
    def aiOnnxMl(self):
        version = self.opsets["ai.onnx.ml"]
        if version == 1:
            return self.aiOnnxMlOpset1
        else:
            # Need to raise an exception here
            pass

    @property
    def aiGraphcore(self):
        version = self.opsets["ai.graphcore"]
        if version == 1:
            return self.aiGraphcoreOpset1
        else:
            # Need to raise an exception here
            pass
