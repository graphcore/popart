# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import argparse
import glob
import json
import os
import tarfile
import tempfile
import urllib.request

import popart
import numpy as np
import onnx
from onnx import numpy_helper

# Onnx modelzoo models are hosted on GitHub as tarballs, with URL:
# https://github.com/onnx/models/blob/master/vision/classification/<MODEL_NAME>/model/<MODEL_NAME>.tar.gz
# Example:
# https://github.com/onnx/models/blob/master/vision/classification/resnet/model/resnet101-v1-7.tar.gz
#
# Here we test that we can load these popular models into the Popart Ir and test against the
# given input / output data:
# 1. Download the tarball to /tmp/modelzoo/<temporary_folder>
# 2. Tarball extracts to /tmp/modelzoo/<temporary_folder>/<archived_folder_name>
# 3. Onnx model path is /tmp/modelzoo/<temporary_folder>/<archived_folder_name>/model.onnx
# 4. Read onnx proto into a Popart Session
# 5. Create the Popart Ir
# 6. Get the output and compare tensors against the downloaded output

# Get download url and test number from args

def is_json_load_ok(report):
    is_report_ok = True
    try:
        report_json = json.loads(report)
    except ValueError:
        is_report_ok = False
    return is_report_ok, report_json

def save_report(zoo_test_dir, execution_report, graph_report):
    if (not os.path.exists(zoo_test_dir)):
        print("Creating directory %s" % (zoo_test_dir))
        os.mkdir(zoo_test_dir)

    is_execution_report_ok, execution_report_json = is_json_load_ok(
        execution_report)
    is_graph_report_ok, graph_report_json = is_json_load_ok(graph_report)

    if is_execution_report_ok:
        execution_report_file = os.path.join(zoo_test_dir,
                                             "execution_report.json")
        with open(execution_report_file, 'w') as f:
            json.dump(execution_report_json, f)

    if is_graph_report_ok:
        graph_report_file = os.path.join(zoo_test_dir, "graph_report.json")
        with open(graph_report_file, 'w') as f:
            json.dump(graph_report_json, f)

def total_execution_cycles(execution_report):
    is_execution_report_ok, execution_report_json = is_json_load_ok(
        execution_report)

    if is_execution_report_ok and 'simulation' in execution_report_json:
        if 'cycles' in execution_report_json['simulation']:
            print('cycles: ', execution_report_json['simulation']['cycles'])

        if 'steps' in execution_report_json['simulation']:
            sum_cycles = sum(
                step['cycles']
                for step in execution_report_json['simulation']['steps']
                if 'cycles' in step)
            print("sum_cycles: ", sum_cycles)

def total_tile_sizes(graph_report):
    is_graph_report_ok, graph_report_json = is_json_load_ok(graph_report)
    
    if is_graph_report_ok and 'memory' in graph_report_json and \
        'byTile' in graph_report_json['memory'] and \
        'total' in graph_report_json['memory']['byTile']:
        sum_tile_sizes = sum(graph_report_json['memory']['byTile']['total'])
        print('sum_tile_sizes: ', sum_tile_sizes)

parser = argparse.ArgumentParser()
parser.add_argument("url", type=str, help="URL for the tar file download")
parser.add_argument("test_number",
                    type=int,
                    help="test number to run against, usually in [0,3)")
parser.add_argument("--input_tensor",
                    required=False,
                    type=str,
                    help="Specify the input tensor ID, "
                    "sometimes required if tensors are not in order.")
parser.add_argument("--output_tensor",
                    required=False,
                    type=str,
                    help="Specify the input tensor ID, "
                    "sometimes required if tensors are not in order.")
parser.add_argument("--model_zoo_test_dir",
                    required=False,
                    type=str,
                    help="Specify directory for execution and graph reports.")
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

modelzoo_dir = os.path.abspath(os.path.join(tmpdir, 'modelzoo'))
if (not os.path.exists(modelzoo_dir)):
    print("Creating directory %s" % (modelzoo_dir))
    os.mkdir(modelzoo_dir)

modeldir_obj = tempfile.TemporaryDirectory(dir=modelzoo_dir)
modeldir = modeldir_obj.name

# Download and extract
fn = url.split('/')[-1]
download_path = os.path.join(modeldir, fn)
print("Download path:", download_path)
if (not os.path.exists(download_path)):
    print("Downloading model from %s" % (url))
    urllib.request.urlretrieve(url, download_path)
    print("Download complete:", download_path)

# # Get onnx model from extracted tar
with tarfile.open(download_path) as tar:
    extract_path = tar.extractall(path=modeldir)

print(f"The model was extracted to: {modeldir}")

filenames = glob.glob(os.path.join(modeldir, "**"), recursive=True)
onnx_model = [f for f in filenames if ".onnx" in f][-1]
onnx_model = os.path.join(modeldir, onnx_model)
print("ONNX model:", onnx_model)
test_data_dir = os.path.join(modeldir, test_data_dir)

try:
    os.path.exists(onnx_model)
except ValueError:
    print("Onnx model path: ", onnx_model, " doesn't exist")

# Load input tensor for given test number
inputs = []
input_files = [
    f for f in filenames if f"test_data_set_{test_number}/input_0.pb" in f
]
for i in input_files:
    input_file = os.path.join(modeldir, i)
    tensor = onnx.TensorProto()
    with open(input_file, 'rb') as f:
        tensor.ParseFromString(f.read())
    inputs.append(numpy_helper.to_array(tensor))

# Load reference outputs for given test number
ref_outputs = []
ref_output_files = [
    f for f in filenames if f"test_data_set_{test_number}/output_0.pb" in f
]
for i in ref_output_files:
    output_file = os.path.join(modeldir, i)
    tensor = onnx.TensorProto()
    with open(output_file, 'rb') as f:
        tensor.ParseFromString(f.read())
    ref_outputs.append(numpy_helper.to_array(tensor))

# create graph transformer using .onnx file.
builder = popart.Builder(onnx_model)
# Use builder to get input / output tensor ids, or user specified if provided
if args.input_tensor:
    input_ = args.input_tensor
else:
    input_ = builder.getInputTensorIds()[0]
if args.output_tensor:
    output = args.output_tensor
else:
    output = builder.getOutputTensorIds()[0]

print("Input:", input_, "Output:", output)
graph_transformer = popart.GraphTransformer(onnx_model)
graph_transformer.convertAllFixedPointInitializersToConstants()

# Create forward pass session
session = popart.InferenceSession(
    fnModel=graph_transformer.getModelProto(),
    dataFlow=popart.DataFlow(1, {output: popart.AnchorReturnType("All")}),
    deviceInfo=popart.DeviceManager().createIpuModelDevice({}))

# Compile graph
print("Compiling...")
session.prepareDevice()

# Create buffers to receive results from the execution
inferenceAnchors = session.initAnchorArrays()
stepio = popart.PyStepIO({input_: inputs[0]}, inferenceAnchors)

# Run the inference graph
session.run(stepio)

total_execution_cycles(session.getExecutionReport())
total_tile_sizes(session.getGraphReport())

if args.model_zoo_test_dir:
    save_report(args.model_zoo_test_dir, session.getExecutionReport(), 
                session.getGraphReport())

# Check the output from the test data is approximately equal to our inference
try:
    np.testing.assert_almost_equal(ref_outputs[0],
                                   inferenceAnchors[output],
                                   decimal=4)
    print("SUCCESS - Output tensors approximately equal")
except Exception as e:
    print(e)
