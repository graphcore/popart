import poponnx
import numpy as np


# A wrapper around the Session cpp class, renamed SessionCore in pybind,
# to enable more pythonic use. See session.hpp for parameter descriptions
class Session(poponnx.SessionCore):
    def __init__(self,
                 fnModel,
                 earlyInfo,
                 dataFeed,
                 losses=[],
                 optimizer=None,
                 outputdir="",
                 passes=[],
                 cTens=[],
                 userOptions=poponnx.SessionOptionsCore()):
        super(Session,
              self).__init__(fnModel, earlyInfo, dataFeed, losses, optimizer,
                             cTens, outputdir, userOptions, passes)
        self.dataFeed = dataFeed

    def initAnchorArrays(self):
        anchorArrays = {}
        for anchor in self.dataFeed.anchors():
            x = self.getInfo(anchor)
            outShape = x.shape()
            # Note : == is not the same as "is" here.
            dfArt = self.dataFeed.art()
            batchesPerStep = self.dataFeed.batchesPerStep()
            if dfArt == poponnx.AnchorReturnType.ALL:
                outShape[0] = outShape[0] * batchesPerStep
            elif dfArt == poponnx.AnchorReturnType.SUM:
                outShape[0] = outShape[0] / batchesPerStep
            elif dfArt == poponnx.AnchorReturnType.FINAL:
                outShape[0] = outShape[0]
            else:
                raise RuntimeError("unrecognised AnchorType")
            anchorArrays[anchor] = np.empty(
                shape=outShape, dtype=x.data_type_lcase())

        return anchorArrays
