# Copyright (c) 2018 Graphcore Ltd. All rights reserved.
"""
Framework independent functionality for driving PopART
"""


class NetWriter():
    """
    Base class, to be inherited once per framework

    Arguments:
        inNames:
            A list (in order) of all the inputs to the ONNX Model.
        outNames:
            names of the outputs of the ONNX Model.
        optimizer:
            An optimizer (ConstSGD, SGD, etc) or ``None`` if
            in inference mode.
        anchors:
            Only relevant if in training mode: the names of tensors
            which must be computed and returned. If not in training
            mode, then outputs of forward are the (only) tensors
            to return.
        dataFeed:
            Configuration for the data feeds and fetches.
        inputShapeInfo:
            For every loss stream input and standard input: the shape,
            ONNX DataType and how to get data.

    """

    def __init__(self, inNames, outNames, optimizer, dataFeed, inputShapeInfo):

        self.inNames = inNames
        self.outNames = outNames
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
        Perform ``batchesPerStep`` training steps. This function
        only needs to be implemented by frameworks which will
        be used to verify PopART. See ``torchwriter.py`` for an
        example implementation.
        """
        raise NotImplementedError()

    def infer(self, inputsMap):
        """
        Perform ``batchesPerStep`` inference steps. This function
        only needs to be implemented by frameworks which will
        be used to verify PopART. See ``torchwriter.py`` for an
        example implementation.
        """
        raise NotImplementedError()
