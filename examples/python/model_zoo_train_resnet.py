import argparse
import glob
import os
import tarfile
import tempfile
import urllib.request

import numpy as np
from PIL import Image
import onnx
import poponnx
from onnx import numpy_helper

# Onnx modelzoo models are hosted on AWS as tarballs, with URL:
# https://s3.amazonaws.com/download.onnx/models/opset_<VER>/<MODEL_NAME>.tar.gz
# Some specific examples have non-standard urls e.g. resnet-18 URL is:
# https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v1/resnet18v1.tar.gz
# You will need a folder of sample images from the imagenet dataset, along with
# the txt file of image classes here : https://github.com/onnx/models/blob/master/models/image_classification/synset.txt
# Both the directory of the images and the txt file of the classes will need to be specified as arguments
# Here we test that we can load these popular models into the Poponnx Ir and and
# train on some simple data, just to see if it runs.
# 1. Download the tarball to /tmp/modelzoo/
# 2. Tarball extracts to /tmp/modelzoo/<MODEL_NAME>
# 3. Onnx model path is /tmp/modelzoo/<MODEL_NAME>/model.onnx
# 4. Read onnx proto into a Poponnx Session
# 5. Create the Poponnx Ir
# 6. Train against the supplied images


def preprocess(img_data):
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
        norm_img_data[i, :, :] = (
            img_data[i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data


# Get args
parser = argparse.ArgumentParser(
    description="Resnet onnx model zoo training example.")
parser.add_argument("url", help="URL for the onnx model zoo input.")
parser.add_argument(
    "image_directory",
    help="Directory where the imagenet examples are stored.")
parser.add_argument("labels_file", help="File listing imagenet classes.")
parser.add_argument(
    "onnx_opset_version", help="ai.onnx opset version", default=7)
args = parser.parse_args()

url = args.url
image_directory = args.image_directory
labels_file = args.labels_file
onnx_opset_version = int(args.onnx_opset_version)

batches_per_step = 10
# Check URL is valid
try:
    urllib.request.urlopen(url)
except urllib.request.HTTPError:
    print("URL does not exist: ", url)

# Check image directory path is valid
try:
    os.path.exists(image_directory)
except ValueError:
    print("Image directory: ", image_directory, " doesn't exist")

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
try:
    onnx_model = glob.glob(os.path.join(unzipped_path, "*.onnx"))[0]
    os.path.exists(onnx_model)
except ValueError:
    print("Onnx model path: ", onnx_model, " doesn't exist")

# Get the labels and parse to dict
try:
    os.path.exists(labels_file)
except ValueError:
    print("Labels path: ", labels_file, " doesn't exist")

labels_lookup = {}
with open(labels_file, "r") as f:
    for i, line in enumerate(f):
        labels_lookup[line.split(" ")[0]] = {
            "index": i,
            "label": line.replace("\n", "").split(" ")[1:]
        }

# Get all the images and labels
inputs = []
labels = []
files = glob.glob(os.path.join(image_directory, "*.JPEG"))
for f in files:
    # Load image in RGB format - will give H x W x 3 array
    # Some images are grayscale so we must enforce RGB
    img = Image.open(f).convert('RGB')
    img_class = os.path.basename(f).split("_")[0]
    img.load()
    # Resize and convert to array
    img = img.resize((224, 224))
    data = np.asarray(img, dtype="int32")
    img.close()
    # Preprocess as per the provided mean and std dev
    data = preprocess(np.transpose(data, (2, 0, 1)))
    labels.append(
        np.array([labels_lookup[img_class]["index"]], dtype=np.int32))
    # Add a new dimension for images per batch. In this model, this remains
    # as 1 per batch.
    data = data[np.newaxis, :]
    inputs.append(data)

steps_per_epoch = len(inputs) // batches_per_step

# create graph transformer using .onnx file. Use builder to get input / output tensor ids
builder = poponnx.Builder(onnx_model, opsets={"ai.onnx": onnx_opset_version})
input_ = builder.getInputTensorIds()[0]
output = builder.getOutputTensorIds()[0]

# Add a softmax layer as per resnet examples
probs = builder.aiOnnx.softmax([output])

# Add the labels input - onnx model doesn't include inputs for training
lbl_shape = poponnx.TensorInfo("INT32", [1])
lb = builder.addInputTensor(lbl_shape)
graph_transformer = poponnx.GraphTransformer(builder.getModelProto())
graph_transformer.convertAllFixedPointInitializersToConstants()
graph_transformer.prepareNodesForTraining()

# Create the training session and input the Nll Loss function.
trainingOptions = poponnx.SessionOptions()
trainingSession = poponnx.TrainingSession(
    fnModel=graph_transformer.getModelProto(),
    dataFeed=poponnx.DataFlow(batches_per_step,
                              {output: poponnx.AnchorReturnType("ALL")}),
    losses=[poponnx.NllLoss(probs, lb, "loss")],
    optimizer=poponnx.ConstSGD(0.001),
    userOptions=trainingOptions,
    deviceInfo=poponnx.DeviceManager().createIpuModelDevice({}))

# Compile graph
trainingSession.prepareDevice()
trainingSession.weightsFromHost()

# Create buffers to receive results from the execution
trainingAnchors = trainingSession.initAnchorArrays()

for epoch in range(4):
    print("Epoch {} ...".format(epoch))
    j = 0
    for i in range(steps_per_epoch):
        print("Step {} ...".format(i))
        # Input tensor shape (10,1,3,224,224) for 1 image per batch, 10 batches per step.
        trainingStepio = poponnx.PyStepIO(
            {
                input_: np.stack(inputs[j:j + batches_per_step], axis=0),
                lb: np.stack(labels[j:j + batches_per_step], axis=0)
            }, trainingAnchors)
        # Run the training graph
        trainingSession.run(trainingStepio)
        j += batches_per_step

    # Copy the weights to the host from the device
trainingSession.weightsToHost()
