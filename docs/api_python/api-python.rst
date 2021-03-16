PopART Python API
=================

Sessions
--------

.. automodule:: popart.session
    :members:
    :undoc-members:
    :show-inheritance:

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

Session
-------

.. autoclass:: popart_core._InferenceSessionCore
    :members:

.. autoclass:: popart_core._TrainingSessionCore
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
###

.. autoclass:: popart_core.SGD
    :members:
    :undoc-members:
    :private-members:

ConstSGD
########

.. autoclass:: popart_core.ConstSGD
    :members:
    :undoc-members:
    :private-members:

Adam
####

.. autoclass:: popart_core.Adam
    :members:
    :undoc-members:
    :private-members:
