.. _popart_importing:

Importing graphs
----------------

The PopART ``Session`` class creates the runtime environment for executing graphs on IPU
hardware. It can read an ONNX graph from a serialised ONNX model protobuf
(ModelProto), either directly from a file or from memory. A session object can be
constructed either as an ``InferenceSession`` or a ``TrainingSession``

Some metadata must be supplied to augment the data present in the ONNX graph in order to run it,
as described below.

In the following example of importing a graph for inference, TorchVision
is used to create a pre-trained AlexNet graph, with a 4 x 3 x 244 x 244 input. The
graph has an ONNX output called ``out``, and the ``DataFlow`` object
contains an entry to fetch that anchor.

.. literalinclude:: python_examples/importing_graphs.py
  :language: python


The DataFlow object is described in more detail in :any:`popart_executing`.

Creating a session
~~~~~~~~~~~~~~~~~~

The ``Session`` class takes the name of a protobuf file, or the protobuf
itself.  It also takes a ``DataFlow`` object which has information about
how to execute the graph:
  * The number of times to conduct a forward pass (and a backward pass,
    if training) of the graph on the IPU before returning to the host for
    more data.
  * The names of the tensors in the graph used to return the results to the host.

In some ONNX graphs, the sizes of input tensors might not be specified.
In this case, the ``inputShapeInfo`` parameter can be used to specify the
input shapes.  The Poplar framework uses statically allocated memory buffers
and so it needs to know the size of tensors before the compilation.

The ``patterns`` parameter allows the user to select a set of graph transformation
patterns which will be applied to the graph.  Without this parameter, a default
set of optimisation transformations will be applied.

Other parameters to the ``Session`` object are used when you are training the
network instead of performing inference. They describe the types of loss to apply to
the network and the optimiser to use.

An example of creating a session object from an ONNX model is shown below.

.. literalinclude:: python_examples/importing_session.py
  :language: python


In this example, when the ``Session`` object is asked to train the graph, an Nll
loss node will be added to the end of the graph, and a ``ConstSGD`` optimiser
will be used to optimise the parameters in the network.

Session control options
~~~~~~~~~~~~~~~~~~~~~~~

The ``userOptions`` parameter passes options to the session. The available options
are listed in the `PopART C++ API Reference
<https://www.graphcore.ai/docs/popart-c-api-reference>`_.
As well as options to control specific features of
the PopART session, there are also some that allow you to pass options to the underlying
Poplar functions:

* ``engineOptions`` passes options to the Poplar ``Engine`` object created to run the graph.
* ``convolutionOptions`` passes options to the PopLibs convolution functions.
* ``reportOptions`` Controls the instrumentation and generation of profiling information.

See :any:`popart_profiling` for examples of using some of these options.

Full details of the Poplar options can be found in the
`Poplar and PopLibs API Reference
<https://www.graphcore.ai/docs/poplar-api-reference>`_.
