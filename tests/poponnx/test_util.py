import fnmatch
import re
import poponnx
import numpy as np


def get_poplar_cpu_device():

    return poponnx.DeviceManager().createCpuDevice()


def get_ipu_model(compileIPUCode=True, numIPUs=1, tilesPerIPU=1216):

    options = {
        "compileIPUCode": compileIPUCode,
        'numIPUs': numIPUs,
        "tilesPerIPU": tilesPerIPU
    }
    return poponnx.DeviceManager().createIpuModelDevice(options)


def get_compute_sets_from_report(report):

    lines = report.split('\n')
    cs = [x for x in lines if re.search(r' OnTileExecute .*:', x)]
    cs = [":".join(x.split(":")[1:]) for x in cs]
    cs = [x.strip() for x in cs]
    return set(cs)


def check_whitelist_entries_in_compute_sets(cs_list, whitelist):

    result = True
    fail_list = []
    wl = [x + '*' for x in whitelist]
    for cs in cs_list:
        if len([x for x in wl if fnmatch.fnmatch(cs, x)]) == 0:
            fail_list += [cs]
            result = False
    if not result:
        print("Failed to match " + str(fail_list))
    return result


def check_compute_sets_in_whitelist_entries(cs_list, whitelist):

    result = True
    fail_list = []
    wl = [x + '*' for x in whitelist]
    for x in wl:
        if len([cs for cs in cs_list if fnmatch.fnmatch(cs, x)]) == 0:
            fail_list += [x]
            result = False
    if not result:
        print("Failed to match " + str(fail_list))
    return result


def check_all_compute_sets_and_list(cs_list, whitelist):

    return (check_whitelist_entries_in_compute_sets(cs_list, whitelist)
            and check_compute_sets_in_whitelist_entries(cs_list, whitelist))


def get_compute_set_regex_count(regex, cs_list):

    return len([cs for cs in cs_list if re.search(regex, cs)])


class BasicSession:
    def __init__(self, logging_dir):
        self.builder = poponnx.Builder()
        self.early_info = poponnx.InputShapeInfo()
        self._setup_opts(logging_dir)
        self.passes = []
        self.inputs = {}

    def _setup_opts(self, logging_dir):
        self.opts = poponnx.SessionOptionsCore()
        self.opts.logDir = str(logging_dir)

    def add_input_tensor(self, data):
        shape = poponnx.TensorInfo(data)

        tensor_id = self.builder.addInputTensor(shape)
        self.early_info.add(tensor_id, shape)
        self.inputs[tensor_id] = data

        return tensor_id

    def run(self, output, anchors, step_method):
        anchorDefs = {}
        for anchorId in anchors:
            anchorDefs[anchorId] = poponnx.AnchorReturnType("ALL")
        dataFlow = poponnx.DataFlow(1, anchorDefs)
        optimizer = poponnx.ConstSGD(0.01)
        losses = [poponnx.L1Loss(output, "l1LossVal", 0.1)]
        proto = self.builder.getModelProto()

        if step_method == 'infer':
            session = poponnx.InferenceSession(
                fnModel=proto,
                inputShapeInfo=self.early_info,
                dataFeed=dataFlow,
                passes=poponnx.Patterns(self.passes),
                userOptions=self.opts)
        elif step_method == 'train':
            session = poponnx.TrainingSession(
                fnModel=proto,
                inputShapeInfo=self.early_info,
                dataFeed=dataFlow,
                losses=losses,
                optimizer=optimizer,
                passes=poponnx.Patterns(self.passes),
                userOptions=self.opts)

        session.setDevice(get_poplar_cpu_device())
        anchors = session.initAnchorArrays()

        session.prepareDevice()

        stepio = poponnx.PyStepIO(self.inputs, anchors)
        # step method should have a value of 'infer', 'train' or 'evaluate'
        session.run(stepio)

        return anchors
