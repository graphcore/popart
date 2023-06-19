Introduction
============

.. warning::
     The ``popxl`` Python package is currently experimental and may be subject
     to change in future releases in ways that are backwards incompatible
     without warning.

PopXL (usually pronounced "Pop-XL" --- like the t-shirt size) is an experimental Python-based machine learning framework that is
targeted at expert users who have an intimate understanding of Graphcore's IPU
hardware. Due to the experimental nature, it is not currently advised to use
PopXL in customer-facing applications or production environments.

.. note:: An overview of the IPU architecture and programming model can be found in the :doc:`ipu-programmers-guide:index`. For more information on the Poplar graph programming framework, refer to the :doc:`poplar-user-guide:index`.

PopXL allows you to describe machine learning models explicitly, providing
greater flexibility and control over model execution than is possible with other
frameworks. This enables you to fit larger models or larger activations and
to achieve maximum performance. Some of PopXL's key features are:

* Implicit replication.
* Explicit auto-differentiation.
* Explicit loading of tensors to and from Streaming Memory.
* Explicit loading of code to and from Streaming Memory (optional).
* Explicit mapping of operations onto IPUs.
* Explicit scheduling of operations (optional).
* Control over parallelisation of IO and compute operations.
