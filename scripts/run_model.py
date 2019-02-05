import sys
import poponnx

if len(sys.argv) != 2:
    raise RuntimeError("onnx model file name expected as argument")

model_file = sys.argv[1]

opts = poponnx.SessionOptionsCore()
opts.logging = {'all': 'TRACE'}
options = {"compileIPUCode": True, 'numIPUs': 1, "tilesPerIPU": 1216}

# currently, with both Cpu and IpuModel, we have outstanding tasks
# T6384 and T6405, about the conv planner failing (after ~15 mins with Cpu)
device = poponnx.DeviceManager().createCpuDevice()
#createIpuModelDevice(options)

# TODO: change to not use builder when T6675 is complete
builder = poponnx.Builder(model_file)
graph_transformer = poponnx.GraphTransformer(builder.getModelProto())
graph_transformer.convertAllFixedPointInitializersToConstants()

#specific to the task, this output might need changing
output = builder.getOutputTensorIds()[0]
dataFlow = poponnx.DataFlow(1, {output: poponnx.AnchorReturnType("ALL")})

s = poponnx.Session(
    graph_transformer.getModelProto(), dataFeed=dataFlow, userOptions=opts)
s.setDevice(device)
s.prepareDevice()
