import onnx
import os
import onnx.numpy_helper
import torch.onnx
import subprocess
from IPython.core.debugger import Tracer


class Driver:
    def __init__(self, dirname):
        self.dirname = dirname
        # we assume one input, one output for now
        self.fnModel = os.path.join(dirname, "model.onnx")
        self.fnIn = os.path.join(dirname, 'input_0.pb')
        # the path to the executable pydriver, compiled from pydriver.cpp
        self.pydriver_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "pydriver")

    def write_output(self, output):
        """
        output might be (1) a torch.Tensor or (2) a tuple of outputs.
        an example, the output from torch LSTM layer is
        (torch.Tensor, (torch.Tensor, torch.Tensor)).
        We write outputs recursively, depth-first.
        """
        if isinstance(output, tuple):
            for sub_output in output:
                self.write_output(sub_output)

        elif isinstance(output, torch.Tensor):
            fnOut = os.path.join(self.dirname, 'output_%d.pb' % (self.nOut, ))
            output_tensor = onnx.numpy_helper.from_array(
                output.detach().numpy())
            with open(fnOut, 'wb') as f:
                print("Writing output to %s" % (fnOut, ))
                f.write(output_tensor.SerializeToString())

            self.nOut = self.nOut + 1

        else:
            raise RuntimeError("unknown type in write_output")

    def write(self, model, inputs):
        if len(inputs) != 1:
            raise RuntimeError("Only 1 input expected in Driver.write")

        dummy_input = inputs[0]
        model.train()
        for i in range(5):
            dummy_output = model(dummy_input)

        # now jump into eval model.
        model.eval()
        torch.onnx.export(model, dummy_input, self.fnModel, verbose=False)
        dummy_input_tensor = onnx.numpy_helper.from_array(dummy_input.numpy())
        with open(self.fnIn, 'wb') as f:
            f.write(dummy_input_tensor.SerializeToString())

        dummy_output = model(dummy_input)

        self.nOut = 0
        self.write_output(dummy_output)

    def run(self):
        subprocess.call([self.pydriver_path, self.dirname])
