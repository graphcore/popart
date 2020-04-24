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
    builder.addOutputTensor(o)
    loss = popart.L1Loss(o, "l1_loss", 1.0)

    anchor_config = {
        o: popart.AnchorReturnType("ALL"),
        "l1_loss": popart.AnchorReturnType("ALL")
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
                                      losses=[loss],
                                      dataFeed=dataFlow,
                                      deviceInfo=tu.create_test_device(),
                                      userOptions=options)

    session.prepareDevice()
    session.weightsFromHost()

    anchors = session.initAnchorArrays()
    input_a = np.array([1.4], dtype=np.float16)
    stepio = popart.PyStepIO({a: input_a}, anchors)
    session.run(stepio)
    summaryReport = session.getSummaryReport()
    lines = summaryReport.split('\n')
    order = []
    first = False
    countStreams = 0
    countSeq = 0

    # Analyse a sequence:
    # default order :
    #     StreamCopy (FromHost) x2
    #     Add
    #     StreamCopy(ToHost) x2
    #     Absolute
    #     Reduce
    #     StreamCopy(ToHost) x2

    # with the option:
    #     StreamCopy (FromHost) x2
    #     Add
    #     Absolute
    #     Reduce
    #     StreamCopy(ToHost)   x2

    for l in lines:
        if re.search(r"Sequence", l):
            countSeq += 1
            if countSeq >= 7:
                break
        if re.search(r"OnTileExecute: 105/Op/Add", l):
            order.append(1)
            first = True
        if re.search(r"OnTileExecute: 101/abs/Op/Absolute", l):
            order.append(2)
        if re.search(r"101/add/ReduceExpression", l):
            order.append(3)
        if re.search(r"StreamCopy", l) and first:
            order.append(4)
            countStreams += 1
    # The streamcopy to host should only happen at the end (after ReduceExpression)
    # Expected list with the option enabled: [1,2,3,4,4]
    # Expected list without the option: [1,4,4,2,3,4,4]
    assert (order[1] == 2)
    assert (order[2] == 3)
    assert (order[3] == 4)
    # The number of Streamcopies happening in total
    # (start counting from Add) should be 2.
    assert (countStreams == 2)
