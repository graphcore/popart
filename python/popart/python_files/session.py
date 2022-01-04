# Copyright (c) 2018 Graphcore Ltd. All rights reserved.
from typing import Dict, Union

import numpy as np
import os

import popart
from popart_core import _InferenceSessionCore, _TrainingSessionCore


def _initAnchorArrays(sess: Union["InferenceSession", "TrainingSession"]
                      ) -> Dict[str, np.array]:
    """Create the anchor arrays to feed data back into Python with.

    Arguments:
        sess: PopART session.

    Returns:
        Dict of anchor names and their relevant np arrays.
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
        if artId == popart.AnchorReturnTypeId.Final or artId == popart.AnchorReturnTypeId.Sum:
            pass
        elif artId == popart.AnchorReturnTypeId.All:
            anchorArrayShape.insert(0, sess.accumulationFactor)
            anchorArrayShape.insert(0, batchesPerStep)
        elif artId == popart.AnchorReturnTypeId.EveryN:
            if batchesPerStep % sess.dataFlow.art(anchor).rp() != 0:
                raise RuntimeError(
                    "Invalid anchor period, does not divide batchesPerStep")

            arp = sess.dataFlow.art(anchor).rp()
            anchorArrayShape.insert(0, sess.accumulationFactor)
            anchorArrayShape.insert(0, batchesPerStep // arp)

        anchorArrayShape = [x for x in anchorArrayShape if x != 1]
        anchorArrayShape = anchorArrayShape + anchorShape

        anchorArrays[anchor] = np.empty(shape=anchorArrayShape,
                                        dtype=anchorInfo.data_type_lcase())

    return anchorArrays


class OutOfMemoryException(popart.popart_exception):
    def __init__(self, e: popart.popart_exception) -> None:
        """Class initializer

        Arguments:
            e: PopART exception to be thrown.
        """
        super(popart.popart_exception, self).__init__(str(e))
        self.error = e

    def getSummaryReport(self) -> str:
        """Get the summary report

        Returns:
            The summary report string.
        """
        return self.error.getSummaryReport()

    def getProfilePath(self) -> str:
        """Get the absolute path of the profile file (profile.pop)

        Returns:
            The absolute path of profile.pop, or an empty string if not created.
        """
        return self.error.getProfilePath()


def makedirsAndCheckWritable(path):
    os.makedirs(path, exist_ok=True)

    if not os.access(path, os.W_OK):
        raise OSError(f"Unable to write to {path}")


class InferenceSession(_InferenceSessionCore):
    """ Create a runtime class for executing an ONNX graph on a set of IPU
    hardware for inference.

    A wrapper around the ``Session`` C++ class, renamed ``SessionCore`` in pybind,
    to enable more Pythonic use. See ``session.hpp`` for parameter descriptions.

    Arguments:
        fnModel: ONNX model proto. Usually a loaded ONNX model, or from
            ``builder.getModelProto()``.
        dataFlow: Configuration for the data feeds and fetches.
        deviceInfo: ``DeviceInfo`` object specifying device type.
            (one of ``IPU``, ``IPUModel`` or ``CPU``) and count.
        inputShapeInfo: Information about the shapes of input and output
            tensors. Default: ``popart.InputShapeInfo()``.
        patterns: Patterns to be run for optimization etc.
            Note: default for patterns must not be ``popart.Patterns()``.
            When ``import popart`` is run, the default arguments are created.
            If the user then loads a custom pattern using
            ``ctypes.cdll.LoadLibrary(custom_pattern_lib.so)``
            then the already constructed ``popart.Patterns`` will
            not include the custom pattern. Default ``None``.
        userOptions: Session options to apply.
            Default: ``popart.SessionOptions()``.
        name: Session name used in debug to identify this session
            Default: ``inference``
    """

    def __init__(
            self,
            fnModel: bytes,
            dataFlow: Dict[int, Dict],
            deviceInfo: popart.DeviceInfo,
            inputShapeInfo: popart.InputShapeInfo = popart.InputShapeInfo(),
            patterns: popart.Patterns = None,
            userOptions: popart.SessionOptions = popart.SessionOptions(),
            name: str = "inference") -> None:

        if patterns == None:
            patterns = popart.Patterns()

        super(InferenceSession,
              self).__init__(fnModel, dataFlow, deviceInfo, inputShapeInfo,
                             userOptions, patterns, name)

    @property
    def dataFlow(self):
        return self._getDataFlow()

    @property
    def replicationFactor(self):
        return self._replicationFactor()

    @property
    def accumulationFactor(self):
        return self._accumulationFactor()

    def initAnchorArrays(self) -> Dict[str, np.array]:
        """Create the anchor arrays to feed data back into Python with.

        Returns:
            Dict of anchor names and their relevant np arrays.
        """
        return _initAnchorArrays(self)

    def compileAndExport(self, filename) -> None:
        """Compiles the graph and exports it to the specified file.

        This will form the snap::Graph and compile the polar::Executable
        before exporting the executable and metadata.

        Arguments:
            filename: Where to save the executable and metadata. If
                      it does not exist, it will be created.

        Raises:
            popart.OutOfMemoryException: If an out of memory event occurs
            OSError: Thrown in the event of any file system related errors
                     during the export

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

        This will create the ``snap::Graph`` and ``poplar::Engine``, and set up
        ``poplar::Streams``.

        Arguments:
            loadEngine: Load the engine and connect the streams once
                        the device is ready.

        Raises:
            popart.OutOfMemoryException: If an out of memory event occurs
        """

        err = popart.OutOfMemoryError()
        super(InferenceSession, self).prepareDevice(loadEngine, err)

        if not err.isSuccessful():
            raise popart.OutOfMemoryException(err)

    @classmethod
    def fromIr(cls, ir, deviceInfo: popart.DeviceInfo,
               name: str = "fromIr") -> 'InferenceSession':
        self = super().__new__(cls)
        super(InferenceSession, self).__init__(ir=ir,
                                               deviceInfo=deviceInfo,
                                               name=name)

        return self


class TrainingSession(_TrainingSessionCore):
    """Create a runtime class for executing an ONNX graph on a set of IPU
        hardware for training

        A wrapper around the ``Session C++`` class, renamed ``SessionCore`` in pybind,
        to enable more Pythonic use. See ``session.hpp`` for parameter descriptions.

        Arguments:
            fnModel: ONNX model proto. Usually a loaded ONNX model,
                or from ``builder.getModelProto()``.
            dataFlow: Configuration for the data feeds and fetches.
            loss: A TensorId of the final scalar loss to use when training.
            optimizer: The type of optimizer to use when training
                and it's properties.
            deviceInfo: DeviceInfo object specifying device type
                (IPU, IPUModel, CPU) and count.
            inputShapeInfo: Information about the shapes of
                input and output tensors. Default: ``popart.InputShapeInfo()``.
            patterns: Optimization patterns to apply. Default: ``None``.
            userOptions: Session options to apply.
                Default: ``popart.SessionOptions()``.
            name: Session name used in debug to identify this session
                Default: ``training``
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
            name: str = "training") -> None:

        if patterns is None:
            patterns = popart.Patterns()

        super(TrainingSession,
              self).__init__(fnModel, dataFlow, loss, optimizer, deviceInfo,
                             inputShapeInfo, userOptions, patterns, name)

    @property
    def dataFlow(self):
        return self._getDataFlow()

    @property
    def replicationFactor(self):
        return self._replicationFactor()

    @property
    def accumulationFactor(self):
        return self._accumulationFactor()

    def initAnchorArrays(self) -> Dict[str, np.array]:
        """Create the anchor arrays to feed data back into Python with.

        Returns:
            Dict of anchor names and their relevant np arrays.
        """
        return _initAnchorArrays(self)

    def compileAndExport(self, filename) -> None:
        """Compiles the graph and exports it to the specified file.

        This will form the snap::Graph and compile the polar::Executable
        before exporting the executable and metadata.

        Arguments:
            filename: Where to save the executable and metadata. If
                      it does not exist, it will be created.

        Raises:
            popart.OutOfMemoryException: If an out of memory event occurs
            OSError: Thrown in the event of any file system related errors
                     during the export

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

        This will create the ``snap::Graph`` and ``poplar::Engine``, and set up
        ``poplar::Streams``.

        Arguments:
            loadEngine: Load the engine and connect the streams once
                        the device is ready.

        Raises:
            popart.OutOfMemoryException: If an out of memory event occurs
        """

        err = popart.OutOfMemoryError()
        super(TrainingSession, self).prepareDevice(loadEngine, err)

        if not err.isSuccessful():
            raise popart.OutOfMemoryException(err)
