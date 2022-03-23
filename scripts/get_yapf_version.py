# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
"""
A Python program to print the version of yapf.
"""

import sys
import subprocess
out = subprocess.run(["yapf", "--version"], stdout=subprocess.PIPE)

#the string output of running yapf --version
stdoutput = out.stdout.decode('utf-8')

#Assuming there is a "yapf x.y.z" in the output,
#this should be something like yapf 0.27.0
if "yapf" not in stdoutput:
    print(-9)
    sys.exit(-9)

version_string = stdoutput.split("yapf")[-1].strip()
if "." not in version_string:
    print(-99)
    sys.exit(-99)

version_frags = version_string.split(".")
# should be [x, y, z] now
if len(version_frags) is not 3:
    print(-999)
    sys.exit(-999)

version_middle = version_frags[1]
print(version_middle)
