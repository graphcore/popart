Introduction
============

.. warning::
     The ``popxl`` Python package is currently experimental and may be subject
     to change in future releases in ways that are backwards incompatible
     without warning.

.. warning::
  Table :numref:`popxl_support_table` details the level of support PopXL enjoys
  across a number of operating systems and Python versions:

  .. list-table:: PopXL OS/Python version support.
     :widths: 25 25 50
     :header-rows: 1
     :name: popxl_support_table
  
     * - Operating System
       - Python Version
       - Support Status
     * - CentOS 7.6
       - 3.6
       - Limited support
     * - RHEL 8
       - 3.6
       - Limited support
     * - Debian 10
       - 3.7
       - Limited support
     * - **Ubuntu 18.04**
       - **3.6**
       - **Recommended**
     * - Ubuntu 20.04
       - 3.8
       - Limited support

  Other operating systems and Python versions are not currently supported.

PopXL is usually pronounced "Pop-XL" (like the t-shirt size).

PopXL is an experimental Python-based machine learning framework that is
targeted at expert users that have an intimate understanding of Graphcore's
hardware. Due to the experimental nature, it is not currently advised to use
PopXL in customer-facing applications or production environments.

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
