import sys

# We test if we can load pywillow.
# This requires that the python library
# and its dependencies are found, should
# be in the following location if installed:
sys.path.append('../../lib')
sys.path.append('../../python')

# the core library, which wraps C++ willow,
import pywillow

# and some utility python functions.
import writer
