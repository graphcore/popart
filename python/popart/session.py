import popart
import numpy as np

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
        # output
        if batchesPerStep == 1 or artId == popart.AnchorReturnTypeId.FINAL:
            anchorArrayShape = anchorShape
        elif artId == popart.AnchorReturnTypeId.ALL:
            anchorArrayShape = anchorShape
            anchorArrayShape.insert(0, batchesPerStep)
        elif artId == popart.AnchorReturnTypeId.EVERYN:
            anchorArrayShape = anchorShape
            arp = self.dataFeed.art(anchor).rp()
            if arp != batchesPerStep:
                anchorArrayShape.insert(0, batchesPerStep // arp)

        # If the graph replication is enabled and greater than 1 then add
        # an extra dimension for the replication
        if self.replicationFactor > 1:
            anchorArrayShape.insert(0, self.replicationFactor)

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


class InferenceSession(popart.InferenceSessionCore):
    def __init__(self,
                 fnModel,
                 dataFeed,
                 deviceInfo,
                 losses=[],
                 inputShapeInfo=popart.InputShapeInfo(),
                 passes=popart.Patterns(),
                 userOptions=popart.SessionOptionsCore()):

        if passes == None:
            passes = popart.Patterns()

        super(InferenceSession,
              self).__init__(fnModel, dataFeed, deviceInfo, losses,
                             inputShapeInfo, userOptions, passes)

        self.dataFeed = dataFeed
        self.replicationFactor = userOptions.replicatedGraphCount if userOptions.enableReplicatedGraphs else 1

    def initAnchorArrays(self):
        return _initAnchorArrays(self)

    def prepareDevice(self):

        err = popart.PrepareDeviceError()
        super(InferenceSession, self).prepareDevice(err)

        # If an error occured during the perpareDevice raise an exception
        if (not err.isSuccessful()):
            raise popart.PrepareDeviceException(err)


class TrainingSession(popart.TrainingSessionCore):
    def __init__(self,
                 fnModel,
                 dataFeed,
                 losses,
                 optimizer,
                 deviceInfo,
                 inputShapeInfo=popart.InputShapeInfo(),
                 passes=popart.Patterns(),
                 userOptions=popart.SessionOptionsCore()):

        if passes == None:
            passes = popart.Patterns()

        super(TrainingSession,
              self).__init__(fnModel, dataFeed, losses, optimizer, deviceInfo,
                             inputShapeInfo, userOptions, passes)

        self.dataFeed = dataFeed
        self.replicationFactor = userOptions.replicatedGraphCount if userOptions.enableReplicatedGraphs else 1

    def initAnchorArrays(self):
        return _initAnchorArrays(self)

    def prepareDevice(self):

        err = popart.PrepareDeviceError()
        super(TrainingSession, self).prepareDevice(err)

        # If an error occured during the perpareDevice raise an exception
        if (not err.isSuccessful()):
            raise popart.PrepareDeviceException(err)
