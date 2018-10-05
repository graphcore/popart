import numpy as np
import numpy.random as npr

class FromTxtFiles:
    """
    for very small experiments
    """
    def __init__(self, nSamples, batchsize, makeDummy, streams):
        """
        nSamples:
          number of samples in the dataset
        makeDummy:
          if True, generate random data
        streams:
          a dict with keys: tensor names
                      values: a dict with keys:
                               file : the .txt file where the data is
                                      the data is in a single column, 
                                      so 1 element per line
                               type : the type (ONNX DataType string)
                                      of the tensor
                               shape: the shape of the tensor. Note that
                                      this does NOT include batchsize
                                      as a dimension
        """
        self.nSamples = nSamples
        self.streams = streams
        self.batchsize = batchsize
        if makeDummy:
            self.makeDummyData()

        self.data = {}
        self.loadData()

    def loadData(self):
        """
        load the data from the txt files into self.data,
        stored as numpy arrays
        """
        for tensorName in self.streams.keys():
            fileName = self.streams[tensorName]["file"]
            onnxType = self.streams[tensorName]["type"]
            elmShape = self.streams[tensorName]["shape"]
            totShape = [self.nSamples] + elmShape
            dtype_ = None

            if onnxType == "FLOAT":
                dtype_ = np.float32

            elif onnxType == "INT64":
                dtype_ = np.int64

            else:
                raise RuntimeError("unrecognised ONNX type")

            self.data[tensorName] = np.loadtxt(fname = fileName, dtype = dtype_)
            self.data[tensorName] = self.data[tensorName].reshape(totShape)

    def streamNames(self):
        return self.data.keys()

    def streamVals(self):
        return self.data.values()

    def getDataString(self):
        """
        summarise the tensor data, each line will be
        tensorName   ONNX_type   shape_INCLUDING_batchsize
        """
        out = ""
        for tensorName in self.streams.keys():
            onnxType = self.streams[tensorName]["type"]
            elmShape = self.streams[tensorName]["shape"]
            totShape = [self.batchsize] + elmShape
            shape = "(" +  ",".join([str(x) for x in totShape]) + ")"
            out +="%s %s %s\n" % (tensorName, onnxType, shape)
        return out

    def getLoadingString(self):
        out = "FromTxtFiles\n"
        out += "nSamples = %d\n"%(self.nSamples,)
        for tensorName in self.streams.keys():
            fileName = self.streams[tensorName]["file"]
            out +="%s %s\n" % (tensorName, fileName)
        return out


    def makeDummyData(self):
        for tensorName in self.streams.keys():
            fileName = self.streams[tensorName]["file"]
            onnxType = self.streams[tensorName]["type"]
            elmShape = self.streams[tensorName]["shape"]
            totShape = [self.nSamples] + elmShape
            dtype_ = None

            if onnxType == "FLOAT":
                data = npr.randn(*totShape)
                np.savetxt(fname = fileName, X = data.reshape(-1), fmt = "%.3f")

            elif onnxType == "INT64":
                data = npr.randint(low = 0, high = 10, size = totShape)
                np.savetxt(fname = fileName, fmt = "%d", X = data.reshape(-1))

            else:
                raise RuntimeError("unrecognised ONNX type " + onnxType)





