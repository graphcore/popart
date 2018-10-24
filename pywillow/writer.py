"""
Framework independent functionality for driving willow
"""


class NetWriter():
    """
    Base class, to be inherited once per framework
    """

    def __init__(self, inNames, outNames, losses, optimizer, willowVerif,
                 dataFeed, earlyInfo):
        """
        inNames:
          A list (in order) of all the inputs to the ONNX Model.
        outNames:
          names of the outputs of the ONNX Model
        losses:
          a list of willow Loss objects
        optimizer:
          an optimizer (see optimizers.py) or None if
          in evaluation mode
        anchors:
          only relevant if in training mode: the names of Tensors
          which must be computed and returned. If not in training
          mode, then outputs of forward are the (only) tensors
          to return
        willowVerif:
          (if training mode) generate the ONNX models at succesive
          training steps
          (if evaluation mode) to be decided
        dataFeed:
          how to get data
        earlyInfo:
          for every loss stream input and standard input: the shape,
          ONNX DataType and how to get data
        """
        self.inNames = inNames
        self.outNames = outNames
        self.losses = losses
        self.optimizer = optimizer
        self.willowVerif = willowVerif
        self.dataFeed = dataFeed
        self.earlyInfo = earlyInfo
        self.trainMode = optimizer != None

        print(self.dataFeed.nAnchors())

        if ((self.dataFeed.nAnchors() != 0) and (self.trainMode is False)):
            raise RuntimeError("anchors only for trainMode")

    def writeOnnx(self, dirname):
        """
        To be implemented once per framework,
        see torchwriter.py for ideas
        """
        raise NotImplementedError()

    def write(self, dirname):
        """
            writeOnnx : framework specific details of
            generating the ONNX model
        """

        # write remaining, framework specific calls
        self.writeOnnx(dirname)

        print("writer.py has completed the write\n------\n")
