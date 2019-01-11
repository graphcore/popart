import poponnx
import numpy as np


# A wrapper around the Session cpp class, renamed SessionCore in pybind,
# to enable more pythonic use. See session.hpp for parameter descriptions
class Session(poponnx.SessionCore):
    def __init__(self,
                 fnModel,
                 dataFeed,
                 inputShapeInfo=poponnx.InputShapeInfo(),
                 losses=[],
                 optimizer=None,
                 passes=poponnx.Patterns(),
                 userOptions=poponnx.SessionOptionsCore()):

        if passes == None:
            passes = poponnx.Patterns()

        super(Session, self).__init__(fnModel, dataFeed, inputShapeInfo,
                                      losses, optimizer, userOptions, passes)
        self.dataFeed = dataFeed

    def initAnchorArrays(self):
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

            anchorArrays[anchor] = np.empty(
                shape=anchorArrayShape, dtype=anchorInfo.data_type_lcase())

        return anchorArrays
