# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import argparse
import glob
import json
import os
import tarfile
import tempfile
import urllib.request

import numpy as np
from PIL import Image

import popart

desc = """
Usage : python model_zoo_neural_style.py <url>  <input_file.jpg> <output_file.jpg>

Where <url> is one of the style models taken from:
https://github.com/onnx/models/tree/master/vision/style_transfer/fast_neural_style

Click 'Download (with sample test data)' and follow to the download link.

input_file needs to be 224 x 224 pixels, as the models are exported with fixed dimensions.
"""


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


def load_image(image_fn):
    # loading input and resize if needed
    image = Image.open(image_fn)
    size_reduction_factor = 1
    image = image.resize((int(image.size[0] / size_reduction_factor),
                          int(image.size[1] / size_reduction_factor)),
                         Image.ANTIALIAS)
    x = np.array(image).astype('float32')
    x = np.transpose(x, [2, 0, 1])
    x = np.expand_dims(x, axis=0)
    return x


def save_image(result, fn):
    result = np.clip(result, 0, 255)
    result = result.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(result)
    img.save(fn)


def load_onnx_file(url):
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
        tar.extractall(path=modeldir)

    print(f"The model was extracted to: {modeldir}")

    filenames = glob.glob(os.path.join(modeldir, "**"), recursive=True)
    onnx_model = [f for f in filenames if ".onnx" in f][-1]
    onnx_model = os.path.join(modeldir, onnx_model)
    print("ONNX model:", onnx_model)

    try:
        os.path.exists(onnx_model)
    except ValueError:
        print("Onnx model path: ", onnx_model, " doesn't exist")

    return onnx_model, modeldir_obj


def compile_and_run(image_input, image_output, onnx_model):

    img_data = load_image(image_input)

    # create graph transformer using .onnx file.
    print(onnx_model)
    builder = popart.Builder(onnx_model)

    input_ = builder.getInputTensorIds()[0]
    output = builder.getOutputTensorIds()[0]

    print("Input:", input_, "Output:", output)
    graph_transformer = popart.GraphTransformer(onnx_model)
    graph_transformer.convertAllFixedPointInitializersToConstants()

    # Create forward pass session
    session = popart.InferenceSession(
        fnModel=graph_transformer.getModelProto(),
        dataFlow=popart.DataFlow(1, {output: popart.AnchorReturnType("All")}),
        deviceInfo=popart.DeviceManager().acquireAvailableDevice(1))

    # Compile graph
    print("Compiling...")
    session.prepareDevice()

    # Create buffers to receive results from the execution
    inferenceAnchors = session.initAnchorArrays()
    stepio = popart.PyStepIO({input_: img_data.copy()}, inferenceAnchors)

    # Run the inference graph
    session.run(stepio)

    total_execution_cycles(session.getExecutionReport())
    total_tile_sizes(session.getGraphReport())

    data_out = inferenceAnchors[output]

    save_image(np.squeeze(data_out), image_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("url", type=str, help="URL for the tar file download")
    parser.add_argument("image_input", type=str, help="Input image file")
    parser.add_argument("image_output", type=str, help="Output image file")

    args_ = parser.parse_args()

    onnx_model_, modeldir_obj = load_onnx_file(args_.url)
    compile_and_run(args_.image_input, args_.image_output, onnx_model_)
