import os
import sys
from torch.autograd import Variable
import torch
import torch.nn
import torch.nn.functional
import torchvision
sys.path.append("../../driver")
import torchdriver
import importlib
importlib.reload(torchdriver)
import subprocess

import model0
import model1
import model2
import model4


if (len(sys.argv) != 2):
    raise RuntimeError("onnx_net.py <log directory>")

model_number =  0

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
elif model_number == 4:
    nChans = 5
    model = model4.Model4(nChans)
else:
    raise RuntimeError("invalid model number")

model.write(dirname=outputdir)
model.run(dirname=outputdir)


allDotPrefixes = [x[0:-4] for x in os.listdir(outputdir) if ".dot" in x]
print("Will generate graph pdfs for all of:")
print(allDotPrefixes)
for name in allDotPrefixes:
    dotfile = os.path.join(outputdir, "%s.dot"%(name,))
    outputfile = os.path.join(outputdir, "%s.pdf"%(name,))
    log = subprocess.call(["dot", "-T", "pdf", "-o", outputfile, dotfile])
    print("Exit status on `%s' was: %s"%(name, log))
print("torchdriver calling script complete.")
