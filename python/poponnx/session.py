import poponnx
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
        if batchesPerStep == 1 or artId == poponnx.AnchorReturnTypeId.FINAL:
            anchorArrayShape = anchorShape
        elif artId == poponnx.AnchorReturnTypeId.ALL:
            anchorArrayShape = anchorShape
            anchorArrayShape.insert(0, batchesPerStep)
        elif artId == poponnx.AnchorReturnTypeId.EVERYN:
            anchorArrayShape = anchorShape
            arp = self.dataFeed.art(anchor).rp()
            if arp != batchesPerStep:
                anchorArrayShape.insert(0, batchesPerStep // arp)

        # If the graph replication is enabled and greater than 1 then add
        # an extra dimension for the replication
        if self.replicationFactor > 1:
            anchorArrayShape.insert(0, self.replicationFactor)

        anchorArrays[anchor] = np.empty(
            shape=anchorArrayShape, dtype=anchorInfo.data_type_lcase())

    return anchorArrays


class InferenceSession(poponnx.InferenceSessionCore):
    def __init__(self,
                 fnModel,
                 dataFeed,
                 deviceInfo,
                 losses=[],
                 inputShapeInfo=poponnx.InputShapeInfo(),
                 passes=poponnx.Patterns(),
                 userOptions=poponnx.SessionOptionsCore()):

        if passes == None:
            passes = poponnx.Patterns()

        super(InferenceSession,
              self).__init__(fnModel, dataFeed, deviceInfo, losses,
                             inputShapeInfo, userOptions, passes)

        self.dataFeed = dataFeed
        self.replicationFactor = userOptions.replicatedGraphCount if userOptions.enableReplicatedGraphs else 1

    def initAnchorArrays(self):
        return _initAnchorArrays(self)


class TrainingSession(poponnx.TrainingSessionCore):
    def __init__(self,
                 fnModel,
                 dataFeed,
                 losses,
                 optimizer,
                 deviceInfo,
                 inputShapeInfo=poponnx.InputShapeInfo(),
                 passes=poponnx.Patterns(),
                 userOptions=poponnx.SessionOptionsCore()):

        if passes == None:
            passes = poponnx.Patterns()

        super(TrainingSession,
              self).__init__(fnModel, dataFeed, losses, optimizer, deviceInfo,
                             inputShapeInfo, userOptions, passes)

        self.dataFeed = dataFeed
        self.replicationFactor = userOptions.replicatedGraphCount if userOptions.enableReplicatedGraphs else 1

    def initAnchorArrays(self):
        return _initAnchorArrays(self)
