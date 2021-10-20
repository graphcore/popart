# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart
import numpy as np
import test_util as tu
import re


@tu.requires_ipu_model
def test_groupHostSync():
    builder = popart.Builder()

    a = builder.addInputTensor(popart.TensorInfo("FLOAT16", [1]))
    w = builder.addInitializedInputTensor(np.ones([1], np.float16))
    o = builder.aiOnnx.add([w, a])
    l1 = builder.aiGraphcore.l1loss([o], 0.1)

    anchor_config = {
        o: popart.AnchorReturnType("All"),
        l1: popart.AnchorReturnType("All")
    }
    dataFlow = popart.DataFlow(1, anchor_config)

    options = popart.SessionOptions()
    options.engineOptions = {
        "debug.instrumentCompute": "true",
        "debug.instrumentExternalExchange": "true"
    }
    options.groupHostSync = True  #The option we are testing
    options.reportOptions = {
        "showVarStorage": "true",
        "showPerIpuMemoryUsage": "true",
        "showExecutionSteps": "true"
    }
    patterns = popart.Patterns(popart.PatternsLevel.Minimal)

    session = popart.InferenceSession(fnModel=builder.getModelProto(),
                                      dataFlow=dataFlow,
                                      deviceInfo=tu.create_test_device(),
                                      userOptions=options,
                                      patterns=patterns)

    session.prepareDevice()
    session.weightsFromHost()

    anchors = session.initAnchorArrays()
    input_a = np.array([1.4], dtype=np.float16)
    stepio = popart.PyStepIO({a: input_a}, anchors)
    session.run(stepio)
    summaryReport = session.getSummaryReport()
    lines = summaryReport.split('\n')
    order = []
    pastSwitch = False
    countSeq = 0

    # Analyse a sequence:
    # default order :
    #  Switch
    #   Repeat
    #     StreamCopy (FromHost) x1
    #     StreamCopy(ToHost) x1
    #     Add
    #     StreamCopy(ToHost) x2
    #     Absolute
    #     Reduce
    #     StreamCopy(ToHost) x1

    # with the option:
    #  Switch
    #   Repeat
    #     StreamCopy (FromHost) x1
    #     Add
    #     Absolute
    #     Reduce
    #     StreamCopy(ToHost)   x1

    for l in lines:
        if re.search(r"Switch", l):
            pastSwitch = True
        if not pastSwitch:
            continue
        if re.search(r"Sequence", l):
            countSeq += 1
            if countSeq > 6:
                break
        if re.search(r"OnTileExecute: ", l):
            order.append("execution")
        if re.search(r"\bStreamCopy(Mid)?\b", l):
            order.append("streamcopy")

    # The streamcopy to host should only happen at the end (after
    # ReduceExpression)
    # There should be 2 stream copies and some executions
    assert len(order) > 3
    # There should be 1 stream copy at the start and 1 at the end
    assert order[0] == "streamcopy"
    assert order[-1] == "streamcopy"
    # 2 stream copies in total
    assert order.count("streamcopy") == 2
    # Everything else should be execution
    for i in order[2:-1]:
        assert i == 'execution'
