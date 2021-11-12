# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import TYPE_CHECKING

from .add import *
from .call import *
from .cast import *
from .detach import *
from .div import *
from .dropout import *
from .dynamic_slice import *
from .dynamic_update import *
from .gather import *
from .gelu import *
from .group_norm import *
from .host_load import *
from .host_store import *
from .increment_mod import *
from .init import *
from .ipu_copy import *
from .logical_and import *
from .logical_not import *
from .logical_or import *
from .repeat import *
from .matmul import *
from .mul import *
from .negative_log_likelihood import *
from .negate import *
from .random import *
from .print_tensor import *
from .reshape import *
from .remote_load import *
from .remote_store import *
from .scaled_add import *
from .scatter import *
from .slice import *
from .softmax import *
from .split import *
from .squeeze import *
from .sub import *
from .transpose import *

import popart.ir.ops.collectives
import popart.ir.ops.var_updates

if TYPE_CHECKING:
    # Static Type Checking requires "import .. as"
    # however this causes a circular import at runtime
    import popart.ir.ops.collectives as collectives
    import popart.ir.ops.var_updates as var_updates
