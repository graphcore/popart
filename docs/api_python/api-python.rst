PopART Python API
=================

Sessions
--------

Training session
^^^^^^^^^^^^^^^^

.. autoclass:: popart.TrainingSession
    :members:
    :undoc-members:


Inference session
^^^^^^^^^^^^^^^^^

.. autoclass:: popart.InferenceSession
    :members:
    :undoc-members:

Data input and output
^^^^^^^^^^^^^^^^^^^^^

.. note:: The base class for IO in PopART is `IStepIO`. The way in which this class is used is detailed in our `C++ API documentation <https://docs.graphcore.ai/projects/popart-cpp-api/en/latest/api-cpp.html#data-input-and-output-istepio>`_.

.. autoclass:: popart.PyStepIO
    :special-members: __init__
    :members:

.. autoclass:: popart.PyStepIOCallback
    :special-members: __init__
    :members:

Builder
--------

.. automodule:: popart.builder
    :members:
    :undoc-members:
    :show-inheritance:

Tensor information
------------------

.. automodule:: popart.tensorinfo
    :members:
    :undoc-members:
    :show-inheritance:

Writer
------

.. automodule:: popart.writer
    :members:
    :undoc-members:
    :show-inheritance:

Builder
-------

.. autoclass:: popart_core._BuilderCore
    :members:

AiGraphcoreOpset1
^^^^^^^^^^^^^^^^^

.. autoclass:: popart_core.AiGraphcoreOpset1
    :members:

Patterns
--------

.. autoclass:: popart::Patterns
    :members:

Session Options
---------------

.. autoclass:: popart::AccumulateOuterFragmentSettings
  :members:
.. autoclass:: popart::AccumulateOuterFragmentSchedule

.. autoclass:: popart::AutodiffSettings
  :members:
.. autoclass:: popart::AutodiffStitchStrategy

.. autoclass:: popart::BatchSerializationSettings
  :members:
.. autoclass:: popart::BatchSerializationBatchSchedule

.. autoclass:: popart::DotCheck
  :members:

.. autoclass:: popart::ExecutionPhaseSchedule

.. autoclass:: popart::TensorLocationSettings
  :members:

.. autoclass:: popart::ReplicatedTensorSharding

.. autoclass:: popart::TensorLocation

.. autoclass:: popart::TensorStorage

.. autoclass:: popart::TileSet

.. autoclass:: popart::Instrumentation

.. autoclass:: popart::MergeVarUpdateType

.. autoclass:: popart::SyntheticDataMode

.. autoclass:: popart::RecomputationType

.. autoclass:: popart::VirtualGraphMode

.. autoclass:: popart::SubgraphCopyingStrategy

.. autoclass:: popart::ExecutionPhaseSettings
  :members:

.. autoclass:: popart_core.SessionOptions
    :members:

Optimizers
----------

.. autoclass:: popart_core.Optimizer
    :members:

SGD
^^^

.. autoclass:: popart_core.SGD
    :members:
    :undoc-members:
    :private-members:

ConstSGD
^^^^^^^^

.. autoclass:: popart_core.ConstSGD
    :members:
    :undoc-members:
    :private-members:

Adam
^^^^

.. autoclass:: popart_core.Adam
    :members:
    :undoc-members:
    :private-members:

``popart.ir`` (experimental)
------------------------------

.. warning::
     This Python module is currently experimental and may be subject to change
     in future releases in ways that are backwards incompatible without
     deprecation warnings.

The ``popart.ir`` module is an experimental PopART python module through
which it is possible to create (and to a limited degree manipulate) PopART IRs
directly.

.. automodule:: popart.ir
    :members:
