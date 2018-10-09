"""
Framework independent functionality for driving willow
"""

import os

# We call an executable of willow to run it,
# we have no fancy cython or such likes yet
import subprocess

# Framework independant training class for willow
from optimizers import SGD

class NetWriter():
    """
    Base class, to be inherited once per framework
    """

    def __init__(self, inNames, outNames, losses, optimizer, anchors,
                 willowTest, dataFeed):
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
        willowTest:
          (if training mode) generate the ONNX models at succesive
          training steps
          (if evaluation mode) to be decided
        dataFeed:
          for every loss stream input and standard input: the shape,
          ONNX DataType and how to get data
        """
        self.inNames = inNames
        self.outNames = outNames
        self.losses = losses
        self.optimizer = optimizer
        self.anchors = anchors
        self.willowTest = willowTest
        self.dataFeed = dataFeed
        self.trainMode = optimizer != None

        if ((len(self.anchors) != 0) and (self.trainMode is False)):
            raise RuntimeError("anchors only for trainMode")


    def writeOnnx(self, dirname):
        """
        To be implemented once per framework,
        see torchwriter.py for ideas
        """
        raise NotImplementedError()


    def write(self, dirname):
        """
        (1) creates a file "schedule.txt" describing
            how training should proceed: loading data,
            the optimizer (learning rate etc),
            the batchsizes, etc.

        (2) writeOnnx : framework specific details of
            generating the ONNX model
        """

        # this sectionMarker must match that in pywillow.cpp,
        sectionMarker = ">>>>>>>>"
        schedFn = os.path.join(dirname, "schedule.txt")
        filly = open(schedFn, "w")

        writeSection = lambda s : filly.write("%s %s\n"%(sectionMarker, s))

        writeSection("input names")
        for name in self.inNames:
            filly.write(name)
            filly.write('\n')

        writeSection("output names")
        for name in self.outNames:
            filly.write(name)
            filly.write('\n')

        # write the anchors (the tensors which MUST
        #              be recorded in training mode)
        writeSection("anchor names")
        for name in self.anchors:
            filly.write(name)
            filly.write('\n')

        writeSection("optimizer")
        filly.write(self.optimizer.string())
        filly.write('\n')

        writeSection("log directory")
        filly.write(dirname)
        filly.write('\n')

        writeSection("data info")
        filly.write(self.dataFeed.getDataString())
        filly.write('\n')

        writeSection("data feed")
        filly.write(self.dataFeed.getLoadingString())
        filly.write('\n')

        filly.close()

        # write remaining, framework specific calls
        self.writeOnnx(dirname)

        print("writer.py has completed the write\n------\n")
