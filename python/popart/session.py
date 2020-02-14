from typing import Dict, List, Union

import numpy as np

import popart
from popart_core import _InferenceSessionCore, _TrainingSessionCore


def _initAnchorArrays(sess: Union["InferenceSession", "TrainingSession"]
                      ) -> Dict[str, np.array]:
    """Create the anchor arrays to feed data back into python with.

    Arguments:
        sess {InferenceSession or TrainingSession} -- popart session.

    Returns:
        Dict[str, np.array] --  Dict of anchor names and their relevant np arrays.
    """

    anchorArrays = {}
    for anchor in sess.dataFeed.anchors():
        anchorInfo = sess.getInfo(anchor)

        # Anchor tensor shape corresponds to a single input sample. The
        # anchor array, as returned to the user, has a shape which is a
        # function of sample shape, step size, and return type
        anchorShape = anchorInfo.shape()
        batchesPerStep = sess.dataFeed.batchesPerStep()
        artId = sess.dataFeed.art(anchor).id()

        # There are some conditions where the output is a single sample,
        # and therefore has the same shape as the anchor tensor
        # Otherwise, samples are returned over an extra dimension in the
        # output.
        # With all options enabled return anchors are of the shape:
        # [batches_per_step, accl_factor, repl_factor, micro_batch, *data_shape]

        anchorArrayShape = [sess.replicationFactor]
        if artId == popart.AnchorReturnTypeId.FINAL or artId == popart.AnchorReturnTypeId.SUM:
            pass
        elif artId == popart.AnchorReturnTypeId.ALL:
            anchorArrayShape.insert(0, sess.accumulationFactor)
            anchorArrayShape.insert(0, batchesPerStep)
        elif artId == popart.AnchorReturnTypeId.EVERYN:
            if batchesPerStep % sess.dataFeed.art(anchor).rp() != 0:
                raise RuntimeError(
                    "Invalid anchor period, does not divide batchesPerStep")

            arp = sess.dataFeed.art(anchor).rp()
            anchorArrayShape.insert(0, sess.accumulationFactor)
            anchorArrayShape.insert(0, batchesPerStep // arp)

        anchorArrayShape = [x for x in anchorArrayShape if x != 1]
        anchorArrayShape = anchorArrayShape + anchorShape

        anchorArrays[anchor] = np.empty(shape=anchorArrayShape,
                                        dtype=anchorInfo.data_type_lcase())

    return anchorArrays


class PrepareDeviceException(popart.popart_exception):
    """Custom expection thrown by the devicePrepare call
    """

    def __init__(self, e: popart.popart_exception) -> None:
        """Class initializer

        Arguments:
            e {popart.popart_exception} -- Popart exception to be thrown.
        """
        super(popart.popart_exception, self).__init__(str(e))
        self.error = e

    def getSummaryReport(self) -> str:
        """Get the summary report

        Returns:
            str -- The summary report string.
        """
        return self.error.getSummaryReport()

    def getGraphReport(self) -> str:
        """Get the graph report

        Returns:
            str -- The graph report string.
        """
        return self.error.getGraphReport()


class InferenceSession(_InferenceSessionCore):
    """ Create a runtime class for executing an ONNX graph on a set of IPU
    hardware for inference.

    A wrapper around the Session cpp class, renamed SessionCore in pybind,
    to enable more pythonic use. See session.hpp for parameter descriptions.

    Arguments:
        fnModel {bytes} -- ONNX model proto. Usually a loaded ONNX model, or from
            builder.getModelProto()
        dataFeed {Dict[int, Dict]} -- Configuration for the data feeds and fetches
        deviceInfo {popart.DeviceInfo} -- DeviceInfo object specifying device type
            (IPU, IPUModel, CPU) and count.
        losses {List[popart.Loss]} -- A list of loss layers to use when training (default: {[]})
        inputShapeInfo {popart.InputShapeInfo} -- Information about the shapes of input and output
            tensors (default: {popart.InputShapeInfo()})
        passes {popart.Patterns} -- Patterns to be run for optimization etc.
            Note: default for passes must not be popart.Patterns()
            when `import popart` is run, the default arguments are created.
            If the user then loads a custom pattern via:
            ctypes.cdll.LoadLibrary(custom_pattern_lib.so)
            The already constructed popart.Patterns will
            not include the custom pattern. (default: {None})
        userOptions {popart.SessionOptions} -- Session options to apply. 
            (default: {popart.SessionOptions()})

    Raises:
        popart.PrepareDeviceException: Exception thrown if an invalid device is provided.
    """

    def __init__(
            self,
            fnModel: bytes,
            dataFeed: Dict[int, Dict],
            deviceInfo: popart.DeviceInfo,
            losses: List[popart.Loss] = [],
            inputShapeInfo: popart.InputShapeInfo = popart.InputShapeInfo(),
            passes: popart.Patterns = None,
            userOptions: popart.SessionOptions = popart.SessionOptions()
    ) -> None:

        if passes == None:
            passes = popart.Patterns()

        super(InferenceSession,
              self).__init__(fnModel, dataFeed, deviceInfo, losses,
                             inputShapeInfo, userOptions, passes)

        self.dataFeed = dataFeed
        self.replicationFactor = userOptions.replicatedGraphCount if \
            userOptions.enableReplicatedGraphs else 1
        self.accumulationFactor = userOptions.accumulationFactor if \
            userOptions.enableGradientAccumulation else 1

    def initAnchorArrays(self) -> Dict[str, np.array]:
        """Create the anchor arrays to feed data back into python with.

        Returns:
            Dict[str, np.array] -- Dict of anchor names and their relevant np arrays.
        """
        return _initAnchorArrays(self)

    def prepareDevice(self) -> None:
        """Prepare the network for execution.

        This will create the poplar::Graph, poplar::Engine, and setting up
        poplar::Streams.

        Raises:
            popart.PrepareDeviceException: Exception thrown if an invalid device is provided.
        """

        err = popart.PrepareDeviceError()
        super(InferenceSession, self).prepareDevice(err)

        # If an error occurred during the perpareDevice raise an exception
        if not err.isSuccessful():
            raise popart.PrepareDeviceException(err)


class TrainingSession(_TrainingSessionCore):
    """Create a runtime class for executing an ONNX graph on a set of IPU
        hardware for training

        A wrapper around the Session cpp class, renamed SessionCore in pybind,
        to enable more pythonic use. See session.hpp for parameter descriptions.

        Arguments:
            fnModel {bytes} -- ONNX model proto. Usually a loaded ONNX model,
                or from builder.getModelProto()
            dataFeed {Dict[int, Dict]} -- Configuration for the data feeds and fetches
            losses {List[popart.Loss]} -- A list of loss layers to use when training
            optimizer {popart.Optimizer} -- The type of optimizer to use when training
                and it's properties.
            deviceInfo {popart.DeviceInfo} -- DeviceInfo object specifying device type
                (IPU, IPUModel, CPU) and count.
            inputShapeInfo {popart.InputShapeInfo} -- Information about the shapes of
                input and output tensors (default: {popart.InputShapeInfo()})
            passes {popart.Patterns} -- Optimization patterns to apply (default: {None})
            userOptions {popart.SessionOptions} -- Session options to apply. 
                (default: {popart.SessionOptions()})

        Raises:
            popart.PrepareDeviceException: Exception thrown if an invalid device is provided.
    """

    def __init__(
            self,
            fnModel: bytes,
            dataFeed: Dict[int, Dict],
            losses: List[popart.Loss],
            optimizer: popart.Optimizer,
            deviceInfo: popart.DeviceInfo,
            inputShapeInfo: popart.InputShapeInfo = popart.InputShapeInfo(),
            passes: popart.Patterns = None,
            userOptions: popart.SessionOptions = popart.SessionOptions()
    ) -> None:

        if passes is None:
            passes = popart.Patterns()

        super(TrainingSession,
              self).__init__(fnModel, dataFeed, losses, optimizer, deviceInfo,
                             inputShapeInfo, userOptions, passes)

        self.dataFeed = dataFeed
        self.replicationFactor = userOptions.replicatedGraphCount if \
            userOptions.enableReplicatedGraphs else 1
        self.accumulationFactor = userOptions.accumulationFactor if \
            userOptions.enableGradientAccumulation else 1

    def initAnchorArrays(self) -> Dict[str, np.array]:
        """Create the anchor arrays to feed data back into python with.

        Returns:
            Dict[str, np.array] -- Dict of anchor names and their relevant np arrays.
        """
        return _initAnchorArrays(self)

    def prepareDevice(self) -> None:
        """Prepare the network for execution.

        This will create the poplar::Graph, poplar::Engine, and setting up
        poplar::Streams.

        Raises:
            popart.PrepareDeviceException: Exception thrown if an invalid device is provided.
        """
        err = popart.PrepareDeviceError()
        super(TrainingSession, self).prepareDevice(err)

        # If an error occured during the perpareDevice raise an exception
        if not err.isSuccessful():
            raise popart.PrepareDeviceException(err)
