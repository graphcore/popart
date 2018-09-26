import onnx
import os
import onnx.numpy_helper
import torch.onnx
import subprocess
from IPython.core.debugger import Tracer


# string rules are .: ... : . : ...
# lossName: input1 ... inputN: output : other things specific to the class
class NLL:
    def __init__(self, probId, labelsId):
        self.probId = probId
        self.labelsId = labelsId

    def string(self):
        #TODO : inherit this fuction, : should be common
        return "NLL: %s %s : lossNLL : "%(self.probId, self.labelsId)
   
    def has_stream_in(self):
        return True;

    def stream_string(self, output_names, outputs):
        output_index = output_names.index(self.probId)
        probsShape = outputs[output_index].shape
        batchsize = probsShape[0]
        return "%s %s (%d)"%(self.labelsId, "INT32", batchsize)



class L1:
    def __init__(self, lamb, tensorId):
        self.lamb = lamb
        self.tensorId = tensorId

    def string(self):
        return "L1: %s : lossL1 : %.3f "%(self.tensorId, self.lamb)

    def has_stream_in(self):
        return False


class Driver:
    def __init__(self, dirname):
        self.dirname = dirname
        self.fnModel = os.path.join(dirname, "model.onnx")
        # the path to the executable pydriver, compiled from pydriver.cpp
        self.pydriver_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "pydriver")

    def write_output(self, output, output_names):
        """
        output might be (1) a torch.Tensor or (2) a tuple of outputs.
        an example: the output from torch LSTM layer is
        (torch.Tensor, (torch.Tensor, torch.Tensor)).
        We write outputs recursively, depth-first.
        """
        if isinstance(output, tuple):
            for sub_output in output:
                self.write_output(sub_output, output_names)

        elif isinstance(output, torch.Tensor):
            fnOut = os.path.join(self.dirname, output_names[self.nOut] + ".pb")
            output_tensor = onnx.numpy_helper.from_array(
                output.detach().numpy())
            with open(fnOut, 'wb') as f:
                print("Writing output to %s" % (fnOut, ))
                f.write(output_tensor.SerializeToString())

            self.nOut = self.nOut + 1

        else:
            raise RuntimeError("unknown type in write_output")

    def write(self, model, inputs, input_names, output_names, losses):

        model.train()
        for i in range(5):
            # the star seems to unpack the list,
            dummy_output = model(inputs)



        # now jump into eval model.
        model.eval()
        torch.onnx.export(
            model,
            inputs,
            self.fnModel,
            verbose=False,
            input_names=input_names,
            output_names=output_names)

        # write the inputs
        for i in range(len(input_names)):
            input_i = inputs[i]
            dummy_input_tensor = onnx.numpy_helper.from_array(input_i.numpy())
            input_name_i = input_names[i]
            fn_i = os.path.join(self.dirname, "%s.pb"%(input_name_i,))
            with open(fn_i, 'wb') as f:
                f.write(dummy_input_tensor.SerializeToString())

        dummy_output = model(inputs)

        self.nOut = 0
        self.write_output(dummy_output, output_names)


        # write the input and output names to file
        input_names_fn = os.path.join(self.dirname, "input_names.txt")
        filly = open(input_names_fn, "w")
        for name in input_names:
            filly.write(name)
            filly.write('\n')
        filly.close()

        output_names_fn = os.path.join(self.dirname, "output_names.txt")
        filly = open(output_names_fn, "w")
        for name in output_names:
            filly.write(name)
            filly.write('\n')
        filly.close()

        # write the loss information
        loss_fn = os.path.join(self.dirname, "losses.txt")
        filly = open(loss_fn, "w")
        for loss in losses:
            filly.write(loss.string())
            filly.write('\n')
        filly.close()

        # write the stream-to-loss information
        loss_stream_fn = os.path.join(self.dirname, "loss_stream.txt")
        filly = open(loss_stream_fn, "w")
        for loss in losses:
            if (loss.has_stream_in()):
                filly.write(loss.stream_string(output_names, dummy_output))
        filly.close()

     



    def run(self):
        subprocess.call([self.pydriver_path, self.dirname])
