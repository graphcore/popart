API changes
-----------

The following changes have been made to the PopART API in SDK 1.2. This may
require you to change your code.

Breaking changes
================

.. warning::

    These will require changes to any code that uses them.

Deprecated session options removed
..................................

The deprecated option ``ignoreData`` has been removed.

.. code-block:: python

    # Old
    opts.ignoreData = True

    # New equivalent:
    opts.syntheticDataMode = popart.SyntheticDataMode.Zeros


The deprecated ``enableVirtualGraphs`` and ``autoVirtualGraph`` options have
been removed.

.. code-block:: python

    # Old (manual sharding):
    opts0.enableVirtualGraphs = True
    opts0.autoVirtualGraph = False  # the default

    # New equivalent:
    opts0.virtualGraphMode = popart.VirtualGraphMode.Manual

    # Old (auto sharding):
    opts1.enableVirtualGraphs = True
    opts1.autoVirtualGraph = True

    # New equivalent:
    opts1.virtualGraphMode = popart.VirtualGraphMode.Auto

Optimizer updating API
......................

- ``Session::updateOptimizer`` and ``Session::optimizerFromHost`` have been
  merged into a single call, ``Session::updateOptimizerFromHost``.

- The first call to ``Session::optimizerFromHost`` has been moved inside
  ``Session::prepareDevice``, so it need only be called when updating the
  optimizer from that used to construct the ``Session`` instance.

- Since the methods must be called together anyway in order to have an effect on
  the session, they have been merged to reduce the number of required API calls.

.. code-block:: python

    # Old:
    session.prepareDevice()
    session.optimizerFromHost()
    session.run(stepio)
    session.updateOptimizer(newOpt)
    session.optimizerFromHost()
    session.run(stepio)

    # New equivalent:
    session.prepareDevice()
    session.run(stepio)
    session.updateOptimizerFromHost(newOpt)
    session.run(stepio)

Loss API
........

- The three ``Loss`` types supported by PopART (``L1Loss``, ``NllLoss`` and
  ``IdentityLoss``) are now all exposed to the ``Builder`` interface. This means
  that they will act like other operators. As in torch, users can sum and scale
  the output value of losses to produce a combined loss scalar.

- These losses still take a reduction argument. However, the former
  ``popart.ReductionType.MEAN`` and ``popart.ReductionType.SUM``, have been
  renamed to ``popart.ReductionType.Mean`` and ``popart.ReductionType.Sum``
  respectively (see :ref:`consistent_enum_styles`).

  Previously, the reduction did not actually take place but simply affected the
  gradient calculation. This did not affect training, but was restrictive as it
  could mean that two losses outputting different size tensors could not be
  added together. These reduction arguments now result in a reduction to scalar
  which is consistent with ``mean`` and ``sum`` reduction in PyTorch and allows
  losses of any input shape to be added together.

- There is now an additional reduction type,
  ``popart.ReductionType.NoReduction`` which produces a tensor output similar to
  reduction type ``none`` in PyTorch. For ``L1Loss``, the output is the same
  size as the input, so this is semantically equivalent to ``Abs`` with an
  optional scale parameter.

- The ``InferenceSession`` constructor no longer takes a ``losses`` input
  argument. Any losses you wish to add to your model for the purposes of
  evaluation must be done in the ONNX model.

- The ``TrainingSession`` constructor no longer takes a list of ``popart.Loss`
  instances as a ``losses`` argument. Instead it takes the ``TensorId`` of a
  scalar loss tensor as a (renamed) ``loss`` argument.

- The default ``ReductionType`` for all losses has changed from
  ``ReductionType::Sum`` to ``ReductionType::Mean`` to match that of PyTorch

.. code-block:: python

    ## For an InferenceSession with loss for evaluation

    # Old:
    probs = builder.aiOnnx.softmax([finalActs])
    losses = [popart.NllLoss(probs, labels, "nllLoss")
    session = popart.InferenceSession(losses=losses, dataFeed=popart.DataFlow(1, ["nllLoss"]), ...)

    # New equivalent:
    probs = builder.aiOnnx.softmax([finalActs])
    nll = builder.aiGraphcore.nllloss([probs, label])
    session = popart.InferenceSession(dataFlow=popart.DataFlow(1, [nll]), ...)

    ## For a TrainingSession

    # Old:
    probs = builder.aiOnnx.softmax([finalActs])
    losses = [popart.NllLoss(probs, labels, "nllLoss")  # can optionally reduce to scalar
    session = popart.TrainingSession(losses=losses, ...)

    # New equivalent:
    probs = builder.aiOnnx.softmax([finalActs])
    nll = builder.aiGraphcore.nllloss([probs, label], reduction=popart.ReductionType.Mean)  # must reduce to scalar
    session = popart.TrainingSession(loss=nll, ...)


.. _consistent_enum_styles:

Consistent enumeration styles
.............................

All enums are now PascalCase and some have changed to avoid conflicts with the
Python ``None`` keyword.

+------------------------------------------------------+------------------------------------------------------------+
|                         Old                          |                            New                             |
+======================================================+============================================================+
| ``enum class InitType { NONE = 0, ZERO };``          | ``enum class InitType { NoInit = 0, Zero };``              |
| ``enum class PatternsLevel { NONE, DEFAULT, ALL };`` | ``enum class PatternsLevel { NoPatterns, Default, All };`` |
+------------------------------------------------------+------------------------------------------------------------+


.. code-block:: python

    # Old:
    popart.PatternsLevel.NONE
    popart,InitType.NONE
    popart,InitType.ZERO

    # New equivalent:
    popart.PatternsLevel.NoPatterns
    popart.InitType.NoInit
    popart.InitType.Zero


All other enums have the same name, just with PascalCase, in place of ALLCAPS,
where it wasn't already.


Session constructor argument names
..................................

- ``dataFeed`` has changed to ``dataFlow`` to match the PopART C++ class name.
- ``passes`` has changed to ``patterns`` to match the PopART C++ class name.

.. code-block:: python

    # Old:
    session = popart.TrainingSession(
        dataFeed=popart.DataFlow(1, []),
        passes=popart.Patterns(popart.PatternsLevel.NONE), ...)

    # New equivalent:
    session = popart.TrainingSession(
        dataFlow=popart.DataFlow(1, []),
        patterns=popart.Patterns(popart.PatternsLevel.None), ...)


Exception name change
.....................

``PrepareDeviceException`` is now renamed ``popart.OutOfMemoryException``.

.. code-block:: python

    # Old:
    try:
        session.prepareDevice()
    except popart.PrepareDeviceException as e:
        #handle exception

    # New equivalent:
    try:
        session.prepareDevice()
    except popart.OutOfMemoryException as e:
        #handle exception


Non-breaking changes
====================

These changes are designed to reduce the verbosity of PopART code.

Overloaded DataFlow constructor
...............................


.. code-block:: python

    # Old:
    anchorMap = {
    t0: popart.AnchorReturnType("ALL")
    t1: popart.AnchorReturnType("ALL")
    }
    dataFlow = popart.DataFlow(1, anchorMap)

    # New equivalent:
    dataFlow = popart.DataFlow(1, [t0, t1])

    # Old:
    anchorMap = {
    t0: popart.AnchorReturnType("FINAL")
    t1: popart.AnchorReturnType("FINAL")
    }
    dataFlow = popart.DataFlow(1, anchorMap)

    # New equivalent:
    dataFlow = popart.DataFlow(1, [t0, t1], popart.AnchorReturnType("FINAL"))


Overloaded Builder::addInputTensor method
.........................................

.. code-block:: python

    # Old:
    to_info = popart.TensorInfo("FLOAT", [2, 3, 4])
    t0 = popart.addInputTensor(to_info)

    # New equivalent:
    t0 = popart.addInputTensor("FLOAT", [2, 3, 4])
