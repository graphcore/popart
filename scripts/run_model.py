import sys
import popart

if len(sys.argv) != 2:
    raise RuntimeError("onnx model file name expected as argument")

model_file = sys.argv[1]

opts = popart.SessionOptionsCore()
opts.logging = {'all': 'TRACE'}
options = {"compileIPUCode": True, 'numIPUs': 1, "tilesPerIPU": 1216}

# currently, with both Cpu and IpuModel, we have outstanding tasks
# T6384 and T6405, about the conv planner failing (after ~15 mins with Cpu)
device = popart.DeviceManager().createCpuDevice()
#createIpuModelDevice(options)

# TODO: change to not use builder when T6675 is complete
builder = popart.Builder(model_file)
graph_transformer = popart.GraphTransformer(builder.getModelProto())
graph_transformer.convertAllFixedPointInitializersToConstants()

#specific to the task, this output might need changing
output = builder.getOutputTensorIds()[0]
dataFlow = popart.DataFlow(1, {output: popart.AnchorReturnType("ALL")})

s = popart.Session(graph_transformer.getModelProto(),
                   dataFeed=dataFlow,
                   userOptions=opts)
s.setDevice(device)
s.prepareDevice()
