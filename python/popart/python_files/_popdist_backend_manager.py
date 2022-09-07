# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import threading

_LOCK = threading.Lock()
_REGISTERED = False

_deprecation_msg = (
    "Deprecation warning: "
    "Detected running a multi-instance PopART application. "
    "Automatically initialising the default PopDist backend and registering a "
    "callback to finalise the backend on process exit. In a future release, "
    "this will no longer be done automatically and the user must do it "
    "themselves before creating the Session. You can do this by calling "
    "`popdist.init()`. See the PopDist documentation for more information."
)


def init(print_deprecation: bool = False):
    """
    Call popdist.init(), only the first time this function is called.

    The PopDist library helps with common functionality that involves
    inter-process communication when managing a multi-instance Poplar
    environment.

    The library can dynamically load in any "backend" library for providing the
    lower-level inter-process communication functionality. PopDist ships with its
    own MPI-based backend. This is known as the "default" backend and is the one
    PopART/PopXL uses.

    A PopDist backend must be registered before use. It can only be registered
    once.
    A PopDist backend must be initialised after registration and before use. You
    cannot initialise an already initialised backend.
    A PopDist backend must be finalised after use. You cannot finalise an already
    finalised backend.
    If the user ever does this out of order, it is undefined behaviour, hopefully
    an error.
    At program exit, the backend should be either entirely unregistered, or
    finalised. It is the user's responsibility to ensure this happens.

    Popdist provdes a helper for managing this. It will automatically register
    the default backend, initialise it, and register a callback on program exit
    to finalise it.

    However, we still need to ensure this helper function is only called once.
    In PopXL, we detect that we are running in a multi-instance popdist
    environment during Ir construction, and if so, call this function. A user's
    program may construct many Irs during its lifetime, therefore we need to
    ensure we only call the popdist helper once. That is what this method ensures.

    Args:
        print_deprecation (bool):
            In PopART (not PopXL), the future behaviour is to make the user
            manage the PopDist backend themselves, so we print a deprecation
            warning if they have not already done it before this function gets
            called.
    """

    import popart
    import popdist

    global _LOCK
    global _REGISTERED

    # If not registered, init popdist. This will:
    #  - Register and initialise the default popdist backend
    #  - Register a callback at program exit that will finalise the backend
    with _LOCK:
        if not _REGISTERED:
            if popdist.isBackendInitialized():
                # User already registered a backend manually.
                _REGISTERED = True
            else:
                # Register the default backend for the user.

                if print_deprecation:
                    logger = popart.getLogger()
                    logger.warn(_deprecation_msg)

                popdist.init()
                _REGISTERED = True
