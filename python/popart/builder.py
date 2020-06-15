# Copyright (c) 2018 Graphcore Ltd. All rights reserved.
from typing import Any, Dict, Iterable, List, Union

import numpy as np

import popart
from popart_core import _BuilderCore


class Opset():
    """Minimal base class for the opsets

    Arguments:
        builder: An interface for a Builder, used for creating ONNX graphs.
        version: Opset version to use for the given opset sub-class.
    """
    def __init__(self, builder: "Builder", version: int) -> None:
        self._builder = builder
        self.version = version


class Builder():
    """ A wrapper around the ``Builder`` C++ class, renamed ``BuilderCore`` in pybind,
    to enable more Pythonic use. See ``builder.hpp`` for the class definition.

    Arguments:
        modelProtoOrFilename: Model protobuf string or file path of saved
            ONNX model proto. Default: ``None``.
        opsets: Dict of opset versions. Default: ``None``.
        builderCore: ``_BuilderCore`` object if you want to create a subgraph
            builder using an existing ``buildercore`` object. Default: ``None``.
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
            # but ideally load default opset from opidentifier.hpp
            # See T12084 and T21330
            self.opsets = {"ai.onnx": 10, "ai.onnx.ml": 1, "ai.graphcore": 1}
        else:
            self.opsets = opsets

        # self.aiOnnxMl = AiOnnxMl(self, self.opsets["ai.onnx.ml"])
        self.aiGraphcore = AiGraphcore(self, self.opsets["ai.graphcore"])
        self.aiGraphcoreOpset1 = AiGraphcoreOpset1(self,
                                                   self.opsets["ai.graphcore"])
        self.aiOnnxOpsetVersion(self.opsets["ai.onnx"])

    def aiOnnxOpsetVersion(self, version: int) -> None:
        # Affected by T12084 and T21330
        if f"AiOnnx{version}" in globals():
            self.aiOnnx = globals()[f"AiOnnx{version}"](self, version)
        else:
            raise ValueError(
                f"Unsupported or unrecognized ai.Onnx version: {self.version}")

    def __getattr__(self, name: str) -> Any:
        """Reroute all attribute requests to the underlying ``_BuilderCore`` object

        Arguments:
            name:  attribute required.

        Returns:
            Return value from the ``builder._impl.attr`` call.
        """
        return getattr(self._impl, name)

    def reshape_const(self,
                      aiOnnx: Opset,
                      args: List[str],
                      shape: Iterable[int],
                      debugPrefix: str = "") -> List[int]:
        """Const version of the reshape op.

        Arguments:
            aiOnnx: Versioned aiOnnx opset, for example: ``aiOnnxOpset11``.
            args:  List of tensor ids to feed as arguments.
            shape: Shape to reshape to, for example ``[3, 2, 4]``.

        Keyword Arguments:
            debugPrefix: String to use as a debug prefix. Default: "".

        Returns:
            Output tensor ids.
        """

        newShape = aiOnnx.constant(
            np.array(shape).astype(np.int64), debugPrefix + "_const")
        return aiOnnx.reshape([args[0], newShape], debugPrefix)

    def createSubgraphBuilder(self) -> 'Builder':
        """Create a child builder to add ops to a subgraph using a call operation.

        Returns:
            The child builder.
        """

        subBuilderCore = self._createSubgraphBuilder()
        return Builder(builderCore=subBuilderCore)


class AiOnnx(Opset):
    """Base class for the various AiOnnx builder interfaces. 
    The most recent version of ONNX operators that require 
    special treatment such as Loop, Scan, Logical_If etc. go here.
    While, older versions where the function signature differs
    are implemented on a corresponding subclass.

    Arguments:
        builder: Parent class for access.
        version: ai.Onnx opset version to use; 6 <= version <= 10.
            Default: 10.

    """
    def __init__(self, builder: Builder, version: int) -> None:
        super(AiOnnx, self).__init__(builder, version)

    def __getattr__(self, name: str) -> Any:
        """Reroute all attribute requests to the underlying ``_BuilderCore`` object

        Arguments:
            name: Attribute required.

        Returns:
            Return value of the ``builder._impl.attr`` call.
        """
        return getattr(self.aiOnnx, name)

    def __str__(self):
        return f"AiOnnx{self.version}"

    def logical_if(self,
                   args: List[str],
                   num_outputs: int,
                   else_branch: Builder,
                   then_branch: Builder,
                   name: str = "") -> List[str]:
        """If conditional operation.

        Arguments:
            args: List of tensor ids to feed as arguments.
            num_outputs: Number of output tensors from the if operator.
            else_branch: ``SubgraphBuilder`` for the graph to run if condition
                is false. Has ``num_outputs`` outputs: values you wish to live-out to the subgraph
                created by the if operation, other tensors will not be accessible to the wider graph.
                The number of outputs must match the number of outputs in the ``then_branch``.
            then_branch: ``SubgraphBuilder`` for the graph to run if condition is true.
                Has ``num_outputs`` outputs: values you wish to be live-out to the enclosing scope.
                The number of outputs must match the number of outputs in the ``else_branch``.

        Keyword Arguments:
            name: A string to prepend to the name of the tensor. Default: "".

        Returns:
            Output tensor ids.
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
            args: List of tensor ids to feed as arguments.
            num_outputs:  Number of output tensors from the loop operator.
            body: SubgraphBuilder for the graph to run in the loop.

        Keyword Arguments:
            debugPrefix: A string to prepend to the name of the tensor. Default: "".

        Returns:
            Output tensor ids.
        """
        return self.aiOnnx.loop(args, num_outputs, body._impl, debugPrefix)


class AiOnnx6(AiOnnx):
    """Minimal builder interface for ai.onnx version 6.
    """
    def __init__(self, builder: Builder, version: int) -> None:
        super(AiOnnx6, self).__init__(builder, version)
        self.aiOnnx = self._builder.aiOnnxOpset6


class AiOnnx7(AiOnnx6):
    """Minimal builder interface for ai.onnx version 7.
    """
    def __init__(self, builder: Builder, version: int) -> None:
        super(AiOnnx7, self).__init__(builder, version)
        self.aiOnnx = self._builder.aiOnnxOpset7


class AiOnnx8(AiOnnx7):
    """Minimal builder interface for ai.onnx version 8.
    """
    def __init__(self, builder: Builder, version: int) -> None:
        super(AiOnnx8, self).__init__(builder, version)
        self.aiOnnx = self._builder.aiOnnxOpset8

    def scan(self,
             args: List[str],
             num_outputs: int,
             body: Builder,
             num_scan_inputs: int,
             directions: List[int] = [],
             debugPrefix: str = "") -> List[str]:
        """Scan-8 specific construct op.

        Arguments:
            args: List of tensor ids to feed as arguments.
            num_outputs:  Number of output tensors from the scan operator.
            body: SubgraphBuilder for the graph to run in the scan.
            num_scan_inputs: The number of scan_inputs
            directions: A list of int which specifies the direction
            of the scan_input. 0 indicates forward direction and 1
            indicates reverse direction. If not omitted, scan_input tensors
            will be scanned in the forward direction.

        Keyword Arguments:
            debugPrefix: A string to prepend to the name of the tensor. Default: "".

        Returns:
            Output tensor ids.
        """
        return self.aiOnnx.scan(args, num_outputs, body._impl, num_scan_inputs,
                                directions, debugPrefix)


class AiOnnx9(AiOnnx8):
    """Minimal builder interface for ai.onnx version 9.
    """
    def __init__(self, builder: Builder, version: int) -> None:
        super(AiOnnx9, self).__init__(builder, version)
        self.aiOnnx = self._builder.aiOnnxOpset9

    def scan(self,
             args: List[str],
             num_outputs: int,
             body: Builder,
             num_scan_inputs: int,
             scan_input_axes: List[int] = [],
             scan_input_directions: List[int] = [],
             scan_output_axes: List[int] = [],
             scan_output_directions: List[int] = [],
             debugPrefix: str = "") -> List[str]:
        """Generic Scan construct op.

        Arguments:
            args: List of tensor ids to feed as arguments.
            num_outputs:  Number of output tensors from the scan operator.
            body: SubgraphBuilder for the graph to run in the scan.
            num_scan_inputs: The number of scan_inputs
            scan_input_axes: A list that specifies the axis to be scanned for
                the scan_input. If omitted, 0 will be used as the scan axis for
                every scan_input.
            scan_input_directions: A list that specifies the direction to be
                scanned for the scan_input tensor. 0 indicates forward
                direction and 1 indicates reverse direction.
                If omitted, all scan_input tensors will be scanned in the
                forward direction.
            scan_output_axes: A list that specifies the axis for
                the scan_output. The scan outputs are accumulated
                along the specified axis. If omitted, 0 will be used as
                the scan axis for every scan_output.
            scan_output_directions: A list specifies whether the scan_output
                should be constructed by appending or prepending a new value
                in each iteration: 0 indicates appending and 1 indicates
                prepending. If omitted, all scan_output tensors will be
                produced by appending a value in each iteration.

        Keyword Arguments:
            debugPrefix: A string to prepend to the name of the tensor. Default: "".

        Returns:
            Output tensor ids.
        """
        return self.aiOnnx.scan(args, num_outputs, body._impl, num_scan_inputs,
                                scan_input_axes, scan_input_directions,
                                scan_output_axes, scan_output_directions,
                                debugPrefix)


class AiOnnx10(AiOnnx9):
    """Minimal builder interface for ai.onnx version 10.
    Once ai.onnx version 11 becomes the standard opset,
    this class must be updated to inherit from AiOnnx11, as
    described in T12084
    """
    def __init__(self, builder: Builder, version: int) -> None:
        super(AiOnnx10, self).__init__(builder, version)
        self.aiOnnx = self._builder.aiOnnxOpset10


class AiOnnx11(AiOnnx10):
    """Minimal builder interface for ai.onnx version 11.
    """
    def __init__(self, builder: Builder, version: int) -> None:
        super(AiOnnx11, self).__init__(builder, version)
        self.aiOnnx = self._builder.aiOnnxOpset11


class AiOnnxMl(Opset):
    """Return the builder interface for the given ai.onnx.ml version.

    Raises:
        ValueError: Thrown if an invalid ai.onnx.ml opset version provided.
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
        ValueError: Thrown if an invalid ai.graphcore opset version provided.
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
            args: List of tensor ids to feed as arguments.
            num_outputs: Number of output tensors from the called graph.
            callee: ``SubgraphBuilder`` for the graph to be called.

        Keyword Arguments:
            debugName: A string to prepend to the name of the tensor. Default: "".

        Returns:
            Output tensor ids.
        """
        return self.aiGraphcore.call(args, num_outputs, callee._impl,
                                     debugName)

    def __getattr__(self, name: str) -> Any:
        """Reroute all attribute requests to the underlying ``_BuilderCore`` object

        Arguments:
            name: The name of the attribute to be returned.

        Returns:
            Return value of the ``builder._impl.attr`` call.
        """
        return getattr(self.aiGraphcore, name)


class AiGraphcoreOpset1(AiGraphcore):
    """Sub-class for backwards compatibility. Will forward all calls to AiGraphcore class.
    """
    def __init__(self, builder: Builder, version: int) -> None:
        super(AiGraphcoreOpset1, self).__init__(builder, version)
