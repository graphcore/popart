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

    session = popart.InferenceSession(fnModel=builder.getModelProto(),
                                      dataFlow=dataFlow,
                                      deviceInfo=tu.create_test_device(),
                                      userOptions=options)

    session.prepareDevice()
    session.weightsFromHost()

    anchors = session.initAnchorArrays()
    input_a = np.array([1.4], dtype=np.float16)
    stepio = popart.PyStepIO({a: input_a}, anchors)
    session.run(stepio)
    summaryReport = session.getSummaryReport()
    print(summaryReport)

    lines = summaryReport.split('\n')
    order = []
    pastSwitch = False
    countStreams = 0
    countSeq = 0

    # Analyse a sequence:
    # default order :
    #  Switch
    #   Repeat
    #     StreamCopy (FromHost) x2
    #     StreamCopy(ToHost) x1
    #     Add
    #     StreamCopy(ToHost) x2
    #     Absolute
    #     Reduce
    #     StreamCopy(ToHost) x1

    # with the option:
    #  Switch
    #   Repeat
    #     StreamCopy (FromHost) x2
    #     StreamCopy(ToHost)   x1
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
            if countSeq >= 6:
                break
        if re.search(r"OnTileExecute: 104/Op/Add", l):
            order.append(1)
        if re.search(r"OnTileExecute: 101/abs/Op/Absolute", l):
            order.append(2)
        if re.search(r"101/add/ReduceExpression", l):
            order.append(3)
        if re.search(r"\bStreamCopy\b", l):
            order.append(4)
            countStreams += 1

    # The streamcopy to host should only happen at the end (after
    # ReduceExpression)
    # Expected list with the option enabled: [4,4,4,1,2,3,4]
    # Expected list without the option: [4,4,4,1,4,4,2,3,4]
    assert (order[0] == 4)
    assert (order[1] == 4)
    assert (order[2] == 4)
    assert (order[3] == 1)
    assert (order[4] == 2)
    assert (order[5] == 3)
    assert (order[6] == 4)
    # The number of Streamcopies happening in total
    # (start counting from the Switch) should be 4.
    assert (countStreams == 4)
