import popart
import numpy as np
from popart_core import _InferenceSessionCore, _TrainingSessionCore
# A wrapper around the Session cpp class, renamed SessionCore in pybind,
# to enable more pythonic use. See session.hpp for parameter descriptions


def _initAnchorArrays(self):
    anchorArrays = {}
    for anchor in self.dataFeed.anchors():
        anchorInfo = self.getInfo(anchor)

        # Anchor tensor shape corresponds to a single input sample. The
        # anchor array, as returned to the user, has a shape which is a
        # function of sample shape, step size, and return type
        anchorShape = anchorInfo.shape()
        batchesPerStep = self.dataFeed.batchesPerStep()
        artId = self.dataFeed.art(anchor).id()

        # There are some conditions where the output is a single sample,
        # and therefore has the same shape as the anchor tensor
        # Otherwise, samples are returned over an extra dimension in the
        # output.
        # With all options enabled return anchors are of the shape:
        # [batches_per_step, accl_factor, repl_factor, micro_batch, *data_shape]
        # TODO: T12496 confirm this ordering is correct with a test.

        anchorArrayShape = [self.replicationFactor]
        if artId == popart.AnchorReturnTypeId.FINAL:
            pass
        elif artId == popart.AnchorReturnTypeId.ALL:
            anchorArrayShape.insert(0, self.accumulationFactor)
            anchorArrayShape.insert(0, batchesPerStep)
        elif artId == popart.AnchorReturnTypeId.EVERYN:
            if (batchesPerStep % self.dataFeed.art(anchor).rp() != 0):
                raise RuntimeError(
                    "Invalid anchor period, does not divide batchesPerStep")

            arp = self.dataFeed.art(anchor).rp()
            anchorArrayShape.insert(0, self.accumulationFactor)
            anchorArrayShape.insert(0, batchesPerStep // arp)

        anchorArrayShape = [x for x in anchorArrayShape if x != 1]
        anchorArrayShape = anchorArrayShape + anchorShape

        anchorArrays[anchor] = np.empty(shape=anchorArrayShape,
                                        dtype=anchorInfo.data_type_lcase())

    return anchorArrays


# Custom expection thrown by the devicePrepare call
class PrepareDeviceException(popart.popart_exception):
    def __init__(self, e):
        super(popart.popart_exception, self).__init__(str(e))
        self.error = e

    def getSummaryReport(self):
        return self.error.getSummaryReport()

    def getGraphReport(self):
        return self.error.getGraphReport()


class InferenceSession(_InferenceSessionCore):
    def __init__(
            self,
            fnModel,
            dataFeed,
            deviceInfo,
            losses=[],
            inputShapeInfo=popart.InputShapeInfo(),
            # default for passes must not be popart.Patterns()
            # when `import popart` is run, the default arguments are created.
            # If the user then loads a custom pattern via:
            #     ctypes.cdll.LoadLibrary(custom_pattern_lib.so)
            # The already constructed popart.Patterns will
            # not include the custom pattern.
            passes=None,
            userOptions=popart.SessionOptions()):

        if passes == None:
            passes = popart.Patterns()

        super(InferenceSession,
              self).__init__(fnModel, dataFeed, deviceInfo, losses,
                             inputShapeInfo, userOptions, passes)

        self.dataFeed = dataFeed
        self.replicationFactor = userOptions.replicatedGraphCount if userOptions.enableReplicatedGraphs else 1
        self.accumulationFactor = userOptions.accumulationFactor if userOptions.enableGradientAccumulation else 1

    def initAnchorArrays(self):
        return _initAnchorArrays(self)

    def prepareDevice(self):

        err = popart.PrepareDeviceError()
        super(InferenceSession, self).prepareDevice(err)

        # If an error occurred during the perpareDevice raise an exception
        if (not err.isSuccessful()):
            raise popart.PrepareDeviceException(err)


class TrainingSession(_TrainingSessionCore):
    def __init__(
            self,
            fnModel,
            dataFeed,
            losses,
            optimizer,
            deviceInfo,
            inputShapeInfo=popart.InputShapeInfo(),
            # default for passes must not be popart.Patterns()
            # when `import popart` is run, the default arguments are created.
            # If the user then loads a custom pattern via:
            #     ctypes.cdll.LoadLibrary(custom_pattern_lib.so)
            # The already constructed popart.Patterns will
            # not include the custom pattern.
            passes=None,
            userOptions=popart.SessionOptions()):

        if passes == None:
            passes = popart.Patterns()

        super(TrainingSession,
              self).__init__(fnModel, dataFeed, losses, optimizer, deviceInfo,
                             inputShapeInfo, userOptions, passes)

        self.dataFeed = dataFeed
        self.replicationFactor = userOptions.replicatedGraphCount if userOptions.enableReplicatedGraphs else 1
        self.accumulationFactor = userOptions.accumulationFactor if userOptions.enableGradientAccumulation else 1

    def initAnchorArrays(self):
        return _initAnchorArrays(self)

    def prepareDevice(self):

        err = popart.PrepareDeviceError()
        super(TrainingSession, self).prepareDevice(err)

        # If an error occured during the perpareDevice raise an exception
        if (not err.isSuccessful()):
            raise popart.PrepareDeviceException(err)
