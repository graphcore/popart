import argparse
import numpy as np
import os
import poponnx
import tarfile
import tempfile
import urllib.request

# Onnx modelzoo models are hosted on AWS as tarballs, with URL:
# https://s3.amazonaws.com/download.onnx/models/opset_<VER>/<MODEL_NAME>.tar.gz
#
# Here we test that we can load these popular models into the Poponnx Ir:
# 1. Download the tarball to /tmp/modelzoo/
# 2. Tarball extracts to /tmp/modelzoo/<MODEL_NAME>
# 3. Onnx model path is /tmp/modelzoo/<MODEL_NAME>/model.onnx
# 4. Read onnx proto into a Poponnx Session
# 5. Create the Poponnx Ir)

# Get download url from args
parser = argparse.ArgumentParser()
parser.add_argument("url")
args = parser.parse_args()
url = args.url

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
unzipped_path = os.path.join(modeldir, fn.split('.')[0])
onnx_model = os.path.join(unzipped_path, "model.onnx")
try:
    os.path.exists(onnx_model)
except ValueError:
    print("Onnx model path: ", onnx_model, " doesn't exist")

# TODO: change to not use builder when T6675 is complete
builder = poponnx.Builder(onnx_model)
output = builder.getOutputTensorIds()[0]
graph_transformer = poponnx.GraphTransformer(builder.getModelProto())
graph_transformer.convertAllFixedPointInitializersToConstants()

# Create forward pass session
session = poponnx.InferenceSession(
    fnModel=graph_transformer.getModelProto(),
    dataFeed=poponnx.DataFlow(1, {output: poponnx.AnchorReturnType("ALL")}),
    deviceInfo=poponnx.DeviceManager().createIpuModelDevice({}))

session.prepareDevice()

## For now just forward pass only
