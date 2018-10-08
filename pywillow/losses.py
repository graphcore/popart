# from https://stackoverflow.com/questions/8628123/
#      counting-instances-of-a-class
#      static class variable Python.
from itertools import count

class Loss:
    counter = count(0)

    def __init__(self, name, inputs, output):
        self.inputs = inputs
        self.output = output
        self.name = name

    def string(self):
        """
        string rules are a:  b  :  c  : d
        where
        a = lossName
        b = input1 ... inputN
        c = output
        d = other things specific to the class
            not all losses will have "d"
        """
        return "%s: %s : %s_%d : %s" % (self.name, " ".join(
            self.inputs), self.output, next(self.counter), self.paramsString())

    def paramsString(self):
        """
        see "d" of string(self)
        """
        return ""


class NLL(Loss):
    """
    Negative Log Likelihood loss
    """
    def __init__(self, probId, labelId):
        """
        """
        Loss.__init__(self, "NLL", [probId, labelId], "lossNLL")
        self.probId = probId
        self.labelId = labelId

class L1(Loss):
    """
    L1 loss : lambda * |X|_1
    """
    def __init__(self, lamb, tensorId):
        Loss.__init__(self, "L1", [tensorId], "lossL1")
        self.lamb = lamb
        self.tensorId = tensorId

    def paramsString(self):
        return " %.6f " % (self.lamb, )

