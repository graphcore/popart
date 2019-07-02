import argparse
import glob
import os
import tarfile
import tempfile
import urllib.request

import numpy as np
import onnx
import poponnx
from onnx import numpy_helper

# Onnx modelzoo models are hosted on AWS as tarballs, with URL:
# https://s3.amazonaws.com/download.onnx/models/opset_<VER>/<MODEL_NAME>.tar.gz
# Some specific examples have non-standard urls e.g. resnet-18 URL is:
# https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v1/resnet18v1.tar.gz
#
# Here we test that we can load these popular models into the Poponnx Ir and test against the
# given input / output data:
# 1. Download the tarball to /tmp/modelzoo/
# 2. Tarball extracts to /tmp/modelzoo/<MODEL_NAME>
# 3. Onnx model path is /tmp/modelzoo/<MODEL_NAME>/model.onnx
# 4. Read onnx proto into a Poponnx Session
# 5. Create the Poponnx Ir
# 6. Get the output and compare tensors against the downloaded output

# Get download url and test number from args

parser = argparse.ArgumentParser()
parser.add_argument("url")
parser.add_argument("test_number", type=int)
args = parser.parse_args()

url = args.url
test_number = args.test_number
test_data_dir = 'test_data_set_{}'.format(test_number)
# Check URL is valid
try:
    urllib.request.urlopen(url)
except urllib.request.HTTPError:
    print("URL does not exist: ", url)

# Make base 'modelzoo' dir for downloading and extracting model
tmpdir = tempfile.gettempdir()
modeldir = os.path.abspath(os.path.join(tmpdir, 'modelzoo'))
if (not os.path.exists(modeldir)):
    print("Creating directory %s" % (modeldir))
    os.mkdir(modeldir)

# Download and extract
fn = url.split('/')[-1]
download_path = os.path.join(modeldir, fn)
if (not os.path.exists(download_path)):
    print("Downloading model from %s" % (url))
    urllib.request.urlretrieve(url, os.path.join(modeldir, fn))
    tar = tarfile.open(download_path)
    tar.extractall(path=modeldir)
    tar.close()

# Get onnx model from extracted tar
# TODO: This needs to better handle the different model and tarfile combinations
unzipped_path = os.path.join(modeldir, fn).replace(".tar.gz", "")
test_data_dir = os.path.join(unzipped_path, test_data_dir)
onnx_model = glob.glob(os.path.join(unzipped_path, "*.onnx"))[0]
try:
    os.path.exists(onnx_model)
except ValueError:
    print("Onnx model path: ", onnx_model, " doesn't exist")

# Load input tensor for given test number
inputs = []
inputs_num = len(glob.glob(os.path.join(test_data_dir, 'input_*.pb')))
for i in range(inputs_num):
    input_file = os.path.join(test_data_dir, 'input_{}.pb'.format(i))
    tensor = onnx.TensorProto()
    with open(input_file, 'rb') as f:
        tensor.ParseFromString(f.read())
    inputs.append(numpy_helper.to_array(tensor))

# Load reference outputs for given test number
ref_outputs = []
ref_outputs_num = len(glob.glob(os.path.join(test_data_dir, 'output_*.pb')))
for i in range(ref_outputs_num):
    output_file = os.path.join(test_data_dir, 'output_{}.pb'.format(i))
    tensor = onnx.TensorProto()
    with open(output_file, 'rb') as f:
        tensor.ParseFromString(f.read())
    ref_outputs.append(numpy_helper.to_array(tensor))

# create graph transformer using .onnx file. Use builder to get input / output tensor ids
builder = poponnx.Builder(onnx_model)
input_ = builder.getInputTensorIds()[0]
output = builder.getOutputTensorIds()[0]
graph_transformer = poponnx.GraphTransformer(onnx_model)
graph_transformer.convertAllFixedPointInitializersToConstants()

# Create forward pass session
session = poponnx.InferenceSession(
    fnModel=graph_transformer.getModelProto(),
    dataFeed=poponnx.DataFlow(1, {output: poponnx.AnchorReturnType("ALL")}),
    deviceInfo=poponnx.DeviceManager().createIpuModelDevice({}))

# Compile graph
session.prepareDevice()

# Create buffers to receive results from the execution
inferenceAnchors = session.initAnchorArrays()
stepio = poponnx.PyStepIO({input_: inputs[0]}, inferenceAnchors)

# Run the inference graph
session.run(stepio)

# Check the output from the test data is approximately equal to our inference
try:
    np.testing.assert_almost_equal(ref_outputs[0],
                                   inferenceAnchors[output],
                                   decimal=4)
    print("SUCCESS - Output tensors approximately equal")
except Exception as e:
    print(e)
