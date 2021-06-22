# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir


# NOTE: This class is a placeholder to allow us to test access to
# popart._internal._ir and may not actually be required to implement popart.ir.
# If this class is not required it can be removed.
class Ir:
    """ Class that represents an IR to the popart.ir user. """
    def __init__(self):
        # Member of type popart._internal.ir.Ir that binds to popart::Ir.
        self._ir = _ir.Ir()
