import os
import sys
import model0
import model1
import model2
import model4

import pywillow
from pywillow import Willow

if (len(sys.argv) != 2):
    raise RuntimeError("onnx_net.py <log directory>")

model_number = 0

outputdir = sys.argv[1]
if not os.path.exists(outputdir):
    print("Making %s" % (outputdir, ))
    os.mkdir(outputdir)

writer = None
if model_number == 0:
    nInChans = 20
    nOutChans = 10
    writer = model0.ModelWriter0(nInChans, nOutChans)
elif model_number == 1:
    writer = model1.ModelWriter1()
elif model_number == 2:
    nInChans = 20
    nOutChans = 10
    writer = model2.ModelWriter2(nInChans, nOutChans)
elif model_number == 4:
    nChans = 5
    writer = model4.ModelWriter4(nChans)
else:
    raise RuntimeError("invalid model number")

# write to file(s)
writer.write(dirname=outputdir)
# C++ class reads from file(s) and creates backwards graph
pynet = Willow(outputdir, writer.losses)

allDotPrefixes = [x[0:-4] for x in os.listdir(outputdir) if ".dot" in x]
print("Will generate graph pdfs for all of:")
print(allDotPrefixes)
import subprocess
for name in allDotPrefixes:
    dotfile = os.path.join(outputdir, "%s.dot"%(name,))
    outputfile = os.path.join(outputdir, "%s.pdf"%(name,))
    log = subprocess.call(["dot", "-T", "pdf", "-o", outputfile, dotfile])
    print("Exit status on `%s' was: %s"%(name, log))
print("torchwriter calling script complete.")
