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
                 outputdir="",
                 passes=poponnx.Patterns(),
                 cTens=[],
                 userOptions=poponnx.SessionOptionsCore()):

        if passes == None:
            passes = poponnx.Patterns()

        super(Session,
              self).__init__(fnModel, dataFeed, inputShapeInfo, losses,
                             optimizer, cTens, outputdir, userOptions, passes)
        self.dataFeed = dataFeed

    def initAnchorArrays(self):
        anchorArrays = {}
        for anchor in self.dataFeed.anchors():
            x = self.getInfo(anchor)
            outShape = x.shape()
            # Note : == is not the same as "is" here.
            artId = self.dataFeed.art(anchor).id()
            batchesPerStep = self.dataFeed.batchesPerStep()
            batchSize = self.dataFeed.batchSize()
            if artId == poponnx.AnchorReturnTypeId.ALL:
                outShape[0] = outShape[0] * batchesPerStep
            elif artId == poponnx.AnchorReturnTypeId.EVERYN:
                arp = self.dataFeed.art(anchor).rp()
                outShape[0] = outShape[0] * (batchesPerStep // arp)
            elif artId == poponnx.AnchorReturnTypeId.FINAL:
                outShape[0] = outShape[0]
            else:
                raise RuntimeError("unrecognised AnchorType")
            anchorArrays[anchor] = np.empty(
                shape=outShape, dtype=x.data_type_lcase())

        return anchorArrays
