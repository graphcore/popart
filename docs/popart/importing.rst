.. _popart_importing:

Importing graphs
----------------

The PopART ``Session`` class creates the runtime environment for executing
graphs on IPU hardware. It can read an ONNX graph from a serialised ONNX model
protobuf (ModelProto), either directly from a file or from memory. A ``Session``
object can be constructed either as ``InferenceSession``
(:py:class:`Python <popart.InferenceSession>`, :cpp:class:`C++ <popart::InferenceSession>`) for
inference or ``TrainingSession`` (:py:class:`Python <popart.TrainingSession>`,
:cpp:class:`C++ <popart::Session>`) for training.

Some metadata must be supplied to construct the ``Session`` class. These are described in :numref:`sec_create_session`.

In the following example of importing a graph for inference, the `torchvision package <https://pytorch.org/vision/stable/index.html>`__ is used
to create a pre-trained `AlexNet graph <https://pytorch.org/hub/pytorch_vision_alexnet/>`__ , with a 4 x 3 x 244 x 244 input. The graph
has an ONNX output called ``output``, and the ``DataFlow`` object
(:py:class:`Python <popart.DataFlow>`, :cpp:class:`C++ <popart::DataFlow>`)
contains an entry to fetch that :term:`anchor tensor`.

.. literalinclude:: files/importing_graphs.py
  :language: python

.. only:: html

    :download:`files/importing_graphs.py`

The ``DataFlow`` object is described in more detail in
:numref:`popart_executing`.

.. _sec_create_session:

Creating a session
~~~~~~~~~~~~~~~~~~

The following parameters are required to create a ``Session`` object:

* ``model``: The name of a protobuf file, or the protobuf itself in order to get
  the ONNX graph.
* ``dataFlow``: A ``DataFlow`` object which contains the following needed to
  execute the graph:

  * Batches per step: The number of batches to run in a single call to
    ``Session::run()``:

    * For ``InferenceSession``, this is the number of executions of the model.

    * For ``TrainingSession``, this is the number of weight updates.

  * The names of the tensors in the graph used to return the results to the
    host.

* ``inputShapeInfo``: In some ONNX graphs, the sizes of input tensors might not
  be specified. In this case, the optional ``inputShapeInfo`` parameter can be
  used to specify the input shapes.  The Poplar framework uses statically
  allocated memory buffers and so it needs to know the size of tensors before
  the compilation.

* ``patterns``: The optional ``patterns`` parameter allows the user to select a
  set of graph transformation patterns which will be applied to the graph.
  Without this parameter, a default set of optimisation transformations will be
  applied.

* ``userOptions``: The options to be applied to the session. This is described
  in more detail in :numref:`sec_session_control_options`.

* For ``TrainingSession`` only:

  * ``loss``: The types of loss to apply to the network.
  * ``optimizer``: The optimiser to use.

An example of creating a ``Session`` object from an ONNX model is shown below.

.. literalinclude:: files/importing_session.py
  :language: python

.. only:: html

    :download:`files/importing_session.py`


In this example, when the ``TrainingSession`` object is created, a
negative log likelihood (NLL) loss node (:py:func:`Python <popart.AiGraphcoreOpset1.nllloss>`, :cpp:func:`C++ <popart::AiGraphcoreOpset1::nllloss>`) will be added to the end of the graph,
and a ``ConstSGD`` optimiser will be used to optimise the parameters in the
network.

.. _sec_session_control_options:

Session control options
~~~~~~~~~~~~~~~~~~~~~~~

The optional ``userOptions`` parameter passes options to the session that
control specific features of the PopART session. The available PopART options
are listed in :numref:`session options` in the PopART C++ API
reference.

The ``userOptions`` parameter also controls  the underlying Poplar functions:

* ``engineOptions`` passes options to the Poplar ``Engine`` object created to run the graph.
* ``convolutionOptions`` passes options to the PopLibs convolution functions.
* ``reportOptions`` controls the instrumentation and generation of profiling information.

Full details of the Poplar options can be found in the :doc:`poplar-api:index`.

:numref:`popart_profiling` contains examples of how to use some of these options.

.. _sec_executing_imported_graph:

Executing an imported graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that the device has been selected, the graph can be compiled for it and
loaded onto the hardware. The ``prepareDevice()`` method (:py:func:`Python <popart.TrainingSession.prepareDevice>`, :cpp:func:`C++ <popart::Session::prepareDevice>`) allows you to do this.

In order to execute the graph, input data has to be provided. The input created
here to feed data to PopART must be a NumPy array, rather than an initialised
``torch`` tensor.

Finally, the ``PyStepIO`` class (:py:class:`Python <popart.PyStepIO>`,
:cpp:class:`IStepIO in C++ <popart::IStepIO>`) provides a session with input and
output buffers. For both input and output, this class takes a dictionary with
tensor names as keys and Python (or NumPy) arrays as values. Note that, for the
imported graph, the key should match the tensor names in the graph. In this
case, ``input.1`` is the input tensor name in the imported graph
``alexnet.onnx``, and ``input_1`` is the value fed into it.

In order to find the input names, you can import the ``onnx`` package and use
``onnx.load`` to load the model. ``loaded_model.graph.input`` and
``loaded_model.graph.output`` give you all the node information for inputs and
outputs. You can use the following command to extract the names of inputs and
outputs:

.. code-block:: python

  inputs_name = [node.name for node in loaded_model.graph.input]
  outputs_name = [node.name for node in loaded_model.graph.output]

An example of executing an imported graph is shown below.

.. literalinclude:: files/executing_imported_model.py
  :language: python
