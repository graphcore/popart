"""
Framework independent functionality for driving popart
"""


class NetWriter():
    """
    Base class, to be inherited once per framework
    """

    def __init__(self, inNames, outNames, losses, optimizer, dataFeed,
                 inputShapeInfo):
        """
        inNames:
          A list (in order) of all the inputs to the ONNX Model.
        outNames:
          names of the outputs of the ONNX Model
        losses:
          a list of popart Loss objects
        optimizer:
          an optimizer (see optimizers.py) or None if
          in evaluation mode
        anchors:
          only relevant if in training mode: the names of Tensors
          which must be computed and returned. If not in training
          mode, then outputs of forward are the (only) tensors
          to return
        dataFeed:
          how to get data
        inputShapeInfo:
          for every loss stream input and standard input: the shape,
          ONNX DataType and how to get data
        """
        self.inNames = inNames
        self.outNames = outNames
        self.losses = losses
        self.optimizer = optimizer
        self.dataFeed = dataFeed
        self.inputShapeInfo = inputShapeInfo
        self.trainMode = optimizer != None

        print(self.dataFeed.nAnchors())

    def saveModel(self, filename):
        """
        To be implemented once per framework:
        framework specific details of generating
        the ONNX model and writing it to file
        """
        raise NotImplementedError()

    def train(self, inputsMap):
        """
        perform batchesPerStep training steps. This function
        only needs to be implemented by frameworks which will
        be used to verify popart. See torchwriter.py for an
        example implementation.
        """
        raise NotImplementedError()

    def eval(self, inputsMap):
        """
        perform batchesPerStep evaluation steps. This function
        only needs to be implemented by frameworks which will
        be used to verify popart. See torchwriter.py for an
        example implementation.
        """
        raise NotImplementedError()

    def infer(self, inputsMap):
        """
        perform batchesPerStep inference steps. This function
        only needs to be implemented by frameworks which will
        be used to verify popart. See torchwriter.py for an
        example implementation.
        """
        raise NotImplementedError()
