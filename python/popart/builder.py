from typing import Any, Dict, Iterable, List, Union

import numpy as np

import popart
from popart_core import _BuilderCore


class Opset():
    """Minimal base class for the opsets

    Arguments:
        builder {Builder} -- An interface for a Builder, used for creating ONNX graphs.
        version {int} -- Opset version to use for the given opset sub-class.
    """

    def __init__(self, builder: "Builder", version: int) -> None:
        self._builder = builder
        self.version = version


class Builder():
    """ A wrapper around the Builder cpp class, renamed BuilderCore in pybind,
    to enable more pythonic use. See builder.hpp for the class definition.

    Arguments:
        modelProtoOrFilename {str} -- Model protobuf string or file path of saved
            onnx model proto. (default: {None})
        opsets {dict} -- Dict of opset versions (default: {None})
        builderCore {_BuilderCore} -- _BuilderCore object if wanting to create a subgraph
            builder using an existing buildercore object. (default: {None})
    """

    def __init__(self,
                 modelProtoOrFilename: Union[str, bytes] = None,
                 opsets: Dict[str, int] = None,
                 builderCore: _BuilderCore = None) -> None:
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

    def __getattr__(self, name: str) -> Any:
        """Reroute all attribute requests to the underlying _BuilderCore object

        Arguments:
            name {str} --  attribute required.

        Returns:
            Any --  return of the builder._impl.attr call.
        """
        return getattr(self._impl, name)

    def reshape_const(self,
                      aiOnnx: Opset,
                      args: List[str],
                      shape: Iterable[int],
                      debugPrefix: str = "") -> List[int]:
        """Const version of the reshape op.

        Arguments:
            aiOnnx {Opset} -- versioned aiOnnx opset e.g. aiOnnxOpset11
            args {List[str]} --  List of tensor ids to feed as arguments.
            shape {Iterable[int]} -- shape to rehape to e.g. [3, 2, 4]

        Keyword Arguments:
            debugPrefix {str} -- String to use as a debug prefix. (default: {""})

        Returns:
            List[int] -- Output tensor ids.
        """

        newShape = aiOnnx.constant(
            np.array(shape).astype(np.int64), debugPrefix + "_const")
        return aiOnnx.reshape([args[0], newShape], debugPrefix)

    def createSubgraphBuilder(self) -> 'Builder':
        """Create a child builder to add ops to a subgraph using a call operation.

        Returns:
            Builder -- The child builder.
        """
        subBuilderCore = self._createSubgraphBuilder()
        return Builder(builderCore=subBuilderCore)


class AiOnnx(Opset):
    """Return the builder interface for the given ai.onnx version.

    Arguments:
        builder {Builder} -- parent class for access.
        version {int} -- ai.Onnx opset version to use; 6 < version < 11
            (default 10).

    Raises:
        ValueError: Error thrown if an invalid ai.Onnx opset version provided.
    """

    def __init__(self, builder: Builder, version: int) -> None:
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
            raise ValueError(
                f"Unsupported or unrecognized ai.Onnx version: {self.version}")

    def __getattr__(self, name: str) -> Any:
        """Reroute all attribute requests to the underlying _BuilderCore object

        Arguments:
            name {str} -- Attribute required.

        Returns:
            Any -- return of the builder._impl.attr call.
        """
        return getattr(self.aiOnnx, name)

    def logical_if(self,
                   args: List[str],
                   num_outputs: int,
                   else_branch: Builder,
                   then_branch: Builder,
                   name: str = "") -> List[str]:
        """If conditional operation.

        Arguments:
            args {List[str]} -- List of tensor ids to feed as arguments.
            num_outputs {int} -- Number of output tensors from the if operator.
                else_branch {Builder} -- SubgraphBuilder for the graph to run if condition
                is false. Has num_outputs outputs: values you wish to be live-out to the enclosing 
                scope. The number of outputs must match the number of outputs in the then_branch.
            then_branch {Builder} -- SubgraphBuilder for the graph to run if condition is true.
                Has num_outputs outputs: values you wish to be live-out to the enclosing scope.
                The number of outputs must match the number of outputs in the else_branch.

        Keyword Arguments:
            name {str} -- A string to prepend to the name of the tensor (default: {""})

        Returns:
            List[str] -- Output tensor ids.
        """
        return self.aiOnnx.logical_if(args, num_outputs, else_branch._impl,
                                      then_branch._impl, name)

    def loop(self,
             args: List[str],
             num_outputs: int,
             body: Builder,
             debugPrefix: str = "") -> List[str]:
        """Generic Looping construct op.

        Arguments:
            args {List[str]} -- List of tensor ids to feed as arguments.
            num_outputs {int} --  Number of output tensors from the loop operator.
            body {Builder} -- SubgraphBuilder for the graph to run in the loop.

        Keyword Arguments:
            debugPrefix {str} -- A string to prepend to the name of the tensor (default: {""})

        Returns:
            List[str] -- Output tensor ids.
        """
        return self.aiOnnx.loop(args, num_outputs, body._impl, debugPrefix)


class AiOnnxMl(Opset):
    """Return the builder interface for the given ai.onnx.ml version.

    Raises:
        ValueError: Error thrown if an invalid ai.onnx.ml opset version provided.
    """

    def __init__(self, builder: Builder, version: int) -> None:
        super(AiOnnxMl, self).__init__(builder, version)
        if self.version == 1:
            self.aiOnnxMl = self._builder.aiOnnxMlOpset1
        else:
            raise ValueError(
                f"Unsupported or unrecognized ai.OnnxMl version: {self.version}"
            )

    def __getattr__(self, name) -> Any:
        return getattr(self.aiOnnxMl, name)


class AiGraphcore(Opset):
    """Return the builder interface for the given ai.graphcore version.

    Raises:
        ValueError: Error thrown if an invalid ai.graphcore opset version provided.
    """

    def __init__(self, builder: Builder, version: int) -> None:
        super(AiGraphcore, self).__init__(builder, version)
        if self.version == 1:
            self.aiGraphcore = self._builder.aiGraphcoreOpset1
        else:
            raise ValueError(
                f"Unsupported or unrecognized ai.graphcore version: {self.version}"
            )

    def call(self,
             args: List[int],
             num_outputs: int,
             callee: Builder,
             debugName: str = "") -> List[str]:
        """Add a call operation to the model

        This is a poplar extension, to expose manual code re-use to
        the builder

        Arguments:
            args {List[int]} -- List of tensor ids to feed as arguments.
            num_outputs {int} -- Number of output tensors from the called graph.
            callee {Builder} -- SubgraphBuilder for the graph to be called.

        Keyword Arguments:
            debugName {str} -- A string to prepend to the name of the tensor (default: {""})

        Returns:
            List[str] -- Output tensor ids.
        """
        return self.aiGraphcore.call(args, num_outputs, callee._impl,
                                     debugName)

    def __getattr__(self, name: str) -> Any:
        """Reroute all attribute requests to the underlying _BuilderCore object

        Arguments:
            name {str} --  attribute required.

        Returns:
            Any --  return of the builder._impl.attr call.
        """
        return getattr(self.aiGraphcore, name)


class AiGraphcoreOpset1(AiGraphcore):
    """Sub-class for backwards compatibility. Will forward all calls to AiGraphcore class.
    """

    def __init__(self, builder: Builder, version: int) -> None:
        super(AiGraphcoreOpset1, self).__init__(builder, version)
