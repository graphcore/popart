import os
import sys
from torch.autograd import Variable
import torch.nn
import torch.nn.functional
import torchvision
sys.path.append("../../driver")
import pydriver
import importlib
importlib.reload(pydriver)
import subprocess

import model0
importlib.reload(model0)
import model1
importlib.reload(model1)
import model2
importlib.reload(model2)
import model3
importlib.reload(model3)
import model4
importlib.reload(model4)


if (len(sys.argv) != 2):
    raise RuntimeError("onnx_net.py <log directory>")

model_number = 4

outputdir = sys.argv[1]
if not os.path.exists(outputdir):
    print("Making %s" % (outputdir, ))
    os.mkdir(outputdir)

model = None
if model_number == 0:
    nInChans = 20
    nOutChans = 10
    model = model0.Model0(nInChans, nOutChans)
elif model_number == 1:
    model = model1.Model1()
elif model_number == 2:
    nInChans = 20
    nOutChans = 10
    model = model2.Model2(nInChans, nOutChans)
elif model_number == 3:
    nChans = 25
    model = model3.Model3(nChans)
elif model_number == 4:
    nChans = 5
    model = model4.Model4(nChans)

else:
    raise RuntimeError("invalid model number")

driver = pydriver.Driver(outputdir)
driver.write(
    model,
    inputs=model.inputs,
    input_names=model.input_names,
    output_names=model.output_names,
    anchors=model.anchors,
    losses=model.losses, 
    outputdir=outputdir)
driver.run()

dotfile = os.path.join(outputdir, "jam.dot")
outputfile = os.path.join(outputdir, "jam.pdf")
print("generating %s"%(outputfile,))
#dotgenline = "dot -T -o %s %s"%(outputfile, dotfile,)
log = subprocess.call(["dot", "-T", "pdf", "-o", outputfile, dotfile])
print(log)

print("pydriver python script complete.")
