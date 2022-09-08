# Copyright (c) 2018 Graphcore Ltd. All rights reserved.
from typing import Dict, Union

import numpy as np
import os

import popart
from popart_core import _InferenceSessionCore, _TrainingSessionCore


def _initAnchorArrays(
    sess: Union["InferenceSession", "TrainingSession"]
) -> Dict[str, np.array]:
    """Create the anchor arrays to feed data back into Python.

    Args:
        sess (Union["InferenceSession", "TrainingSession"]): PopART session.

    Raises:
        RuntimeError: If the anchor period is not divisible by
            ``batchesPerStep``.

    Returns:
        Dict[str, np.array]: Dictionary of anchor tensor names and their relevant NumPy arrays.
    """

    anchorArrays = {}
    for anchor in sess.dataFlow.anchors():
        anchorInfo = sess.getInfo(anchor)

        # Anchor tensor shape corresponds to a single input sample. The
        # anchor array, as returned to the user, has a shape which is a
        # function of sample shape, step size, and return type
        anchorShape = anchorInfo.shape()
        batchesPerStep = sess.dataFlow.batchesPerStep()
        artId = sess.dataFlow.art(anchor).id()

        # There are some conditions where the output is a single sample,
        # and therefore has the same shape as the anchor tensor
        # Otherwise, samples are returned over an extra dimension in the
        # output.
        # With all options enabled return anchors are of the shape:
        # [batches_per_step, accl_factor, repl_factor, micro_batch, *data_shape]

        anchorArrayShape = [sess.replicationFactor]
        if (
            artId == popart.AnchorReturnTypeId.Final
            or artId == popart.AnchorReturnTypeId.Sum
        ):
            pass
        elif artId == popart.AnchorReturnTypeId.All:
            anchorArrayShape.insert(0, sess.accumulationFactor)
            anchorArrayShape.insert(0, batchesPerStep)
        elif artId == popart.AnchorReturnTypeId.EveryN:
            if batchesPerStep % sess.dataFlow.art(anchor).rp() != 0:
                raise RuntimeError(
                    "Invalid anchor period, is not divisble by batchesPerStep"
                )

            arp = sess.dataFlow.art(anchor).rp()
            anchorArrayShape.insert(0, sess.accumulationFactor)
            anchorArrayShape.insert(0, batchesPerStep // arp)

        anchorArrayShape = [x for x in anchorArrayShape if x != 1]
        anchorArrayShape = anchorArrayShape + anchorShape

        anchorArrays[anchor] = np.empty(
            shape=anchorArrayShape, dtype=anchorInfo.data_type_lcase()
        )

    return anchorArrays


class OutOfMemoryException(popart.popart_exception):
    """Represent out of memory exceptions that that occur during runtime."""

    def __init__(self, e: popart.popart_exception) -> None:
        """Construct the ``OutOfMemoryException`` class.

        Arguments:
            e: PopART exception to be thrown.
        """
        super(popart.popart_exception, self).__init__(str(e))
        self.error = e

    def getSummaryReport(self) -> str:
        """Get the summary report.

        Returns:
            str: The summary report string.
        """
        return self.error.getSummaryReport()

    def getProfilePath(self) -> str:
        """Get the absolute path of the profile file.

           The profile file is named `profile.pop` and contains full details of
           the exception.

        Returns:
            str: The absolute path of `profile.pop`, or an empty string if the
                file does not exist.
        """
        return self.error.getProfilePath()


def makedirsAndCheckWritable(path):
    os.makedirs(path, exist_ok=True)

    if not os.access(path, os.W_OK):
        raise OSError(f"Unable to write to {path}")


class InferenceSession(_InferenceSessionCore):
    """
    Session for running inference.

    ``InferenceSession`` is a runtime instance that provides an interface for
    executing ONNX graphs on IPU hardware, without any automatic differentiation
    (backpropagation).
    """

    def __init__(
        self,
        fnModel: bytes,
        dataFlow: Dict[int, Dict],
        deviceInfo: popart.DeviceInfo,
        inputShapeInfo: popart.InputShapeInfo = popart.InputShapeInfo(),
        patterns: popart.Patterns = None,
        userOptions: popart.SessionOptions = popart.SessionOptions(),
        name: str = "inference",
    ) -> None:
        """Construct the ``InferenceSession`` class.

        Arguments:
            fnModel: ONNX model proto. Usually a loaded ONNX model, or from
                :py:func:`~popart.builder.getModelProto()`.
            dataFlow: Configuration for the data feeds and fetches.
            deviceInfo: :py:class:`~popart.DeviceInfo` object specifying
                the device type (``IPU``, ``IPUModel`` or ``CPU``) and
                number of each type.
            inputShapeInfo: (Optional) The sizes and dtypes of the input
                tensors. This is used to specify the sizes of the input
                tensors in the case that the ONNX model does not include
                this information. The Poplar graph programmming framework
                uses statically allocated memory buffers and so it needs to
                know the size of tensors before the compilation. Default:
                :py:class:`~popart.InputShapeInfo()`.
            patterns: (Optional) A user-selected set of graph transformation
                patterns which will be applied to the graph. If this is not
                specified, a default set of optimisation transformations
                will be applied. Default ``None``. Note: The default for
                patterns must not be :py:class:`~popart.Patterns()`. When
                ``import popart`` is run, the default arguments are created.
                If the user then loads a custom pattern using
                ``ctypes.cdll.LoadLibrary(custom_pattern_lib.so)`` then the
                already constructed ``popart.Patterns`` will not include the
                custom pattern. Default ``None``.
            userOptions: (Optional) The user configuration options for the
                Session class. Default::py:class:`~popart.SessionOptions()`.
            name: (Optional) The name of this inference session. Default:
                "inference".
        """

        if patterns is None:
            patterns = popart.Patterns()

        super(InferenceSession, self).__init__(
            fnModel, dataFlow, deviceInfo, inputShapeInfo, userOptions, patterns, name
        )

    @property
    def dataFlow(self):
        """Get the configuration for the data feeds and fetches."""
        return self._getDataFlow()

    @property
    def replicationFactor(self):
        """Get the replication factor."""
        return self._replicationFactor()

    @property
    def accumulationFactor(self):
        """Get the gradient accumulation factor."""
        return self._accumulationFactor()

    def initAnchorArrays(self) -> Dict[str, np.array]:
        """Create the anchor arrays to feed data back into Python.

        Returns:
            Dict[str, np.array]: Dictionary of anchor tensor names and their
                relevant NumPy arrays.
        """
        return _initAnchorArrays(self)

    def compileAndExport(self, filename: str) -> None:
        """Compile the graph and export it to a file.

        This method will first create :cpp:class:`snap::Graph` and compile
        :cpp:class:`poplar::Executable`. Next, it will export the executable and
        metadata to the file. The exported file will be in the :doc:`PopEF
        <popef:index>` format. This means that the file can be used to
        run inference using the `Triton Inference Server
        <https://developer.nvidia.com/nvidia-triton-inference-server>`__ with
        the Graphcore Triton backend. See the `Poplar Triton Backend User Guide
        <https://docs.graphcore.ai/projects/poplar-triton-backend/en/latest/index.html>`__
        for more information.

        This method raises an:py:class:`popart.OutOfMemoryException` error if an
        out of memory event occurs. In addition, it raises an ``OSError`` if
        there are any file system related errors.

        Args:
            filename (str): The name of the file where the compiled executable
                and metadata will be saved. If it does not exist, the file will
                be created.

        Raises:
            popart.OutOfMemoryException: If an out of memory event occurs.
            OSError: If there are any file system related errors during the
                export.

        """
        filename = os.path.expanduser(filename)
        if os.path.isdir(filename):
            makedirsAndCheckWritable(filename)
        else:
            makedirsAndCheckWritable(os.path.dirname(filename))

        err = popart.OutOfMemoryError()
        super(InferenceSession, self).compileAndExport(filename, err)

        if not err.isSuccessful():
            raise popart.OutOfMemoryException(err)

    def prepareDevice(self, loadEngine: bool = True) -> None:
        """Prepare the network for execution.

        This will create :cpp:class:`snap::Graph` and
        :cpp:class:`poplar::Engine`, and set up
        :cpp:class:`poplar::Streams`.

        Arguments:
            loadEngine: If ``true``, load the engine and connect the streams
                once the device is ready.

        Raises:
           popart.OutOfMemoryException: If an out of memory event
                occurs.
        """

        err = popart.OutOfMemoryError()
        super(InferenceSession, self).prepareDevice(loadEngine, err)

        if not err.isSuccessful():
            raise popart.OutOfMemoryException(err)

    @classmethod
    def fromIr(
        cls, ir: "Ir", deviceInfo: popart.DeviceInfo, name: str = "fromIr"
    ) -> "InferenceSession":
        """Create a session for inference from an IR.

        Arguments:
            ir (Ir): The IR to create the session from.
            deviceInfo::py:class:`~popart.DeviceInfo` object specifying the
                device type (``IPU``, ``IPUModel`` or ``CPU``) and number of
                each type.
            name (str): The name of this inference session. Default: "fromIr".

        Returns:
            InferenceSession: An inference session.
        """
        self = super().__new__(cls)
        super(InferenceSession, self).__init__(ir=ir, deviceInfo=deviceInfo, name=name)

        return self


class TrainingSession(_TrainingSessionCore):
    """
    Session for training.

    ``TrainingSession`` is a runtime instance that provides an interface for
    executing ONNX graphs on IPU hardware with training provided by optimizing a
    loss tensor using an optimizer and automatic differentiation
    (backpropagation).
    """

    def __init__(
        self,
        fnModel: bytes,
        dataFlow: Dict[int, Dict],
        loss: "",
        optimizer: popart.Optimizer,
        deviceInfo: popart.DeviceInfo,
        inputShapeInfo: popart.InputShapeInfo = popart.InputShapeInfo(),
        patterns: popart.Patterns = None,
        userOptions: popart.SessionOptions = popart.SessionOptions(),
        name: str = "training",
    ) -> None:
        """Construct the ``TrainingSession`` class.

        Arguments:
            fnModel: ONNX model proto. Usually a loaded ONNX model, or from
                :py:func:`~popart.Builder.getModelProto()`.
            dataFlow: Configuration for the data feeds and fetches.
            loss: The identifier of the final scalar loss tensor for
                training.
            optimizer: The name of an optimizer to use when training.
            deviceInfo::py:class:`~popart.DeviceInfo` object specifying the
                device type (``IPU``, ``IPUModel`` or ``CPU``) and number
                of each type.
            inputShapeInfo: (Optional) The sizes and dtypes of the input
                tensors. This is used to specify the sizes of the input
                tensors in the case that the ONNX model does not include
                this information. The Poplar graph programmming framework
                uses statically allocated memory buffers and so it needs to
                know the size of tensors before the compilation. Default:
                :py:class:`~popart.InputShapeInfo()`.
            patterns: (Optional) The optimization patterns to apply.
                Default: ``None``.
            userOptions: The user configuration options for the Session
                class. Default: :py:class:`~popart.SessionOptions()`.
            name: (Optional) The name of this training session. Default:
                ``training``
        """

        if patterns is None:
            patterns = popart.Patterns()

        super(TrainingSession, self).__init__(
            fnModel,
            dataFlow,
            loss,
            optimizer,
            deviceInfo,
            inputShapeInfo,
            userOptions,
            patterns,
            name,
        )

    @property
    def dataFlow(self):
        """Get the configuration for the data feeds and fetches."""
        return self._getDataFlow()

    @property
    def replicationFactor(self):
        """Get the replication factor."""
        return self._replicationFactor()

    @property
    def accumulationFactor(self):
        """Get the gradient accumulation factor."""
        return self._accumulationFactor()

    def initAnchorArrays(self) -> Dict[str, np.array]:
        """Create the anchor arrays to feed data back into Python.

        Returns:
            Dict[str, np.array]: Dictionary of anchor tensor names and their
               relevant NumPy arrays.
        """
        return _initAnchorArrays(self)

    def compileAndExport(self, filename: str) -> None:
        """Compile the graph and export it to a file.

        This method will first create :cpp:class:`snap::Graph` and compile
        :cpp:class:`poplar::Executable`. Next, it will export the executable and
        metadata to the file. The exported file will be in the :doc:`PopEF
        <popef:index>` format. This means that the file can be used to
        run inference using the `Triton Inference Server
        <https://developer.nvidia.com/nvidia-triton-inference-server>`__ with
        the Graphcore Triton backend. See the `Poplar Triton Backend User Guide
        <https://docs.graphcore.ai/projects/poplar-triton-backend/en/latest/index.html>`__
        for more information.

        This method raises an :py:class:`popart.OutOfMemoryException` error if
        an out of memory event occurs. In addition, it raises an ``OSError`` if
        there are any file system related errors.

        Args:
            filename (str): The name of the file where the compiled executable
                and metadata will be saved. If it does not exist, the file will
                be created.

        Raises:
            popart.OutOfMemoryException: If an out of memory event occurs.
            OSError: If there are any file system related errors during the
                export.
        """
        filename = os.path.expanduser(filename)
        if os.path.isdir(filename):
            makedirsAndCheckWritable(filename)
        else:
            makedirsAndCheckWritable(os.path.dirname(filename))

        err = popart.OutOfMemoryError()
        super(TrainingSession, self).compileAndExport(filename, err)

        if not err.isSuccessful():
            raise popart.OutOfMemoryException(err)

    def prepareDevice(self, loadEngine: bool = True) -> None:
        """Prepare the network for execution.

        This will create :cpp:class:`snap::Graph` and
        :cpp:class:`poplar::Engine`, and set up :cpp:class:`poplar::Streams`.

        Arguments:
            loadEngine: If ``true``, load the engine and connect the streams
                once the device is ready.

        Raises:
            popart.OutOfMemoryException: If an out of memory event
                occurs.
        """

        err = popart.OutOfMemoryError()
        super(TrainingSession, self).prepareDevice(loadEngine, err)

        if not err.isSuccessful():
            raise popart.OutOfMemoryException(err)
