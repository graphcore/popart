import popart
import numpy as np
from popart_core import _BuilderCore


class Builder(object):
    """ A wrapper around the Builder cpp class, renamed BuilderCore in pybind,
    to enable more pythonic use. See builder.hpp for the class definition.
    """

    def __init__(self,
                 modelProtoOrFilename=None,
                 opsets=None,
                 builderCore=None):
        """Class initializer

        Arguments:
            modelProtoOrFilename {str} -- Model protobuf string or file path of saved
                onnx model proto. (default: {None})
            opsets {dict} -- Dict of opset versions (default: {None})
            builderCore {_BuilderCore} -- Buildercore object if wanting to create a subgraph
                builder using an existing buildercore object. (default: {None})
        """
        if builderCore is None:
            if modelProtoOrFilename is None:
                self._impl = _BuilderCore()
            else:
                self._impl = _BuilderCore(modelProtoOrFilename)
        else:
            self._impl = builderCore
        if opsets is None:
            # These are the default opsets, they will increment with releases
            self.opsets = {"ai.onnx": 10, "ai.onnx.ml": 1, "ai.graphcore": 1}

            # T12084
            #self.opsets = {"ai.onnx": 11, "ai.onnx.ml": 1, "ai.graphcore": 1}
        else:
            self.opsets = opsets
        self.aiOnnx = AiOnnx(self, self.opsets["ai.onnx"])
        # self.aiOnnxMl = AiOnnxMl(self, self.opsets["ai.onnx.ml"])
        self.aiGraphcore = AiGraphcore(self, self.opsets["ai.graphcore"])
        self.aiGraphcoreOpset1 = AiGraphcoreOpset1(self,
                                                   self.opsets["ai.graphcore"])

    def __getattr__(self, name):
        """Reroute all attribute requests to the underlying _BuilderCore object

        Arguments:
            name {str} -- attribute required.

        Returns:
            various -- return of the builder._impl.attr call.
        """
        return getattr(self._impl, name)

    def reshape_const(self, aiOnnx, args, shape, debugPrefix=""):
        """Const version of the reshape op.

        Arguments:
            aiOnnx {Opset} -- versioned aiOnnx opset e.g. aiOnnxOpset11
            args {list} -- List of tensor ids to feed as arguments.
            shape {list} -- Shape to reshape to.

        Keyword Arguments:
            debugPrefix {str} -- String to use as a debug prefix. (default: {""})

        Returns:
            str -- Output tensor id.
        """
        newShape = aiOnnx.constant(
            np.array(shape).astype(np.int64), debugPrefix + "_const")
        return aiOnnx.reshape([args[0], newShape], debugPrefix)

    def createSubgraphBuilder(self):
        """Create a child builder to add ops to a subgraph using a call operation.

        Returns:
            Builder -- The child builder.
        """
        subBuilderCore = self._createSubgraphBuilder()
        return Builder(builderCore=subBuilderCore)


class Opset(object):
    def __init__(self, builder, version):
        self._builder = builder
        self.version = version


class AiOnnx(Opset):
    def __init__(self, builder, version):
        super(AiOnnx, self).__init__(builder, version)
        if self.version == 11:
            self.aiOnnx = self._builder.aiOnnxOpset11
        elif self.version == 10:
            self.aiOnnx = self._builder.aiOnnxOpset10
        elif self.version == 9:
            self.aiOnnx = self._builder.aiOnnxOpset9
        elif self.version == 8:
            self.aiOnnx = self._builder.aiOnnxOpset8
        elif self.version == 7:
            self.aiOnnx = self._builder.aiOnnxOpset7
        elif self.version == 6:
            self.aiOnnx = self._builder.aiOnnxOpset6
        else:
            # Need to raise an exception here
            pass

    def __getattr__(self, name):
        return getattr(self.aiOnnx, name)

    def logical_if(self, args, num_outputs, else_branch, then_branch, name=""):
        return self.aiOnnx.logical_if(args, num_outputs, else_branch._impl,
                                      then_branch._impl, name)

    def loop(self, args, num_outputs, body, debugPrefix=""):
        return self.aiOnnx.loop(args, num_outputs, body._impl, debugPrefix)


class AiOnnxMl(Opset):
    def __init__(self, builder, version):
        super(AiOnnxMl, self).__init__(builder, version)
        if self.version == 1:
            self.aiOnnxMl = self._builder.aiOnnxMlOpset1
        else:
            # Need to raise an exception here
            pass

    def __getattr__(self, name):
        return getattr(self.aiOnnxMl, name)


class AiGraphcore(Opset):
    def __init__(self, builder, version):
        super(AiGraphcore, self).__init__(builder, version)
        if self.version == 1:
            self.aiGraphcore = self._builder.aiGraphcoreOpset1
        else:
            # Need to raise an exception here
            pass

    def call(self, args, num_outputs, callee, debugName=""):
        return self.aiGraphcore.call(args, num_outputs, callee._impl,
                                     debugName)

    def __getattr__(self, name):
        return getattr(self.aiGraphcore, name)


class AiGraphcoreOpset1(AiGraphcore):
    def __init__(self, builder, version):
        super(AiGraphcoreOpset1, self).__init__(builder, version)
