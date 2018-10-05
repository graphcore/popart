"""
Framework independent functionality for driving neuralnet
"""

import os
import onnx
import onnx.numpy_helper

# We call an executable of neuralnet to run it,
# we have no fancy cython or such likes yet
import subprocess

# Framework independant training class for neuralnet
from losses import NLL, L1
from optimizers import SGD


class OxModule():
    """
    Base class, to be inherited once per framework
    """

    def __init__(self, inNames, outNames, losses, optimizer, anchors,
                 neuralnetTest, dataFeed):
        """
        inNames:
          A list (in order) of all the inputs to the ONNX Model.
        outNames:
          names of the outputs of the ONNX Model
        losses:
          a list of Loss objects (see losses.py)
        optimizer:
          an optimizer (see optimizers.py) or None if
          in evaluation mode
        anchors:
          only relevant if in training mode: the names of Tensors
          which must be computed and returned. If not in training
          mode, then outputs of forward are the (only) tensors
          to return
        neuralnetTest:
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
        self.neuralnetTest = neuralnetTest
        self.dataFeed = dataFeed
        self.trainMode = optimizer != None

        if ((len(self.anchors) != 0) and (self.trainMode is False)):
            raise RuntimeError("anchors only for trainMode")


    def writeOnnx(self, dirname):
        """
        To be implemented once per framework,
        see torchdriver.py for ideas
        """
        raise NotImplementedError()


    def write(self, dirname):
        """
        (1) creates a file "schedule.txt" describing
            how training should proceed: loading data,
            the optimizer (learning rate etc),
            the losses, batchsizes, etc.

        (2) writeOnnx : framework specific details of
            generating the ONNX model
        """

        # this sectionMarker must match that in driver.cpp,
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

        writeSection("losses")
        for loss in self.losses:
            filly.write(loss.string())
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


        print("driver.py has completed the write\n------\n")


    def run(self, dirname):
        # the path to the executable pydriver,
        # compiled from pydriver.cpp. This is the
        # interface to neuralnet
        pydriver_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "driver")
        subprocess.call([pydriver_path, dirname])
