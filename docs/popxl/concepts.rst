Concepts
========

The building blocks of PopXL are the intermediate representation (IR), graphs, tensors and ops. This section describes these concepts. Further information
on how each topic applies to PopXL can be found later in the referenced sections.


IRs
---

An IR is the intermediate representation, in PopART, of the computational graph for a model.

An IR contains one main graph (:numref:`sec_maingraphs`), created on IR initialisation, and multiple subgraphs (:numref:`sec_subgraphs`) that you create. The main components of an IR are shown in :numref:`fig_popxl_building_blocks`.


.. figure:: images/popxl_building_blocks.png
  :width: 90%
  :align: center
  :name: fig_popxl_building_blocks

  An IR contains a main graph (MG) and multiple other graphs (G). Graphs can
  contain ops, intermediate tensors (T) and constant tensors (C). The main
  graph can also contain intermediate, constant and variable tensors (V).


An IR is an executable program that can be run using a PopART session. A Python process can initialise multiple IRs.


.. _graph_concept:

Graphs
------

A graph (:numref:`sec_graphs`) is a computational directed acyclic graph where tensors are edges and ops are nodes. There are two types of graphs: the main graph (:numref:`sec_maingraphs`) and subgraphs (:numref:`sec_subgraphs`). An example is shown in :numref:`fig_popxl_calling_a_graph`.


.. figure:: images/popxl_calling_a_graph.png
   :width: 90%
   :align: center
   :name: fig_popxl_calling_a_graph

   The main graph (MG) calls subgraph 1 (G1) which in turn calls subgraph 2
   (G2). This creates a call tree which is depicted on the right. Op nodes are
   green, intermediate tensors are red and constants are yellow.


* The **main graph** (:numref:`sec_maingraphs`) is the entry point of the IR (like the ``main()`` function in many programming languages). There is only one main graph per IR. The main graph can contain intermediate, constant and variable tensors.

* **Subgraphs** (:numref:`sec_subgraphs`) have input and output tensors. Subgraphs can be called by other graphs using the ``call`` or ``repeat`` op. If a subgraph has multiple call sites, the subgraph is outlined during lowering, leading to code reuse and reduced memory usage. A subgraph can only contain intermediate or constant tensors and not variable tensors. Subgraphs have intermediate tensors which are marked as inputs or outputs. When a subgraph is called, the inputs must be provided by the calling graph. The input data can be either passed by reference or value, and this is determined by the user at the call site.

Tensors
-------

Tensors (:numref:`sec_tensors`) have a shape and data type, and sometimes initialisation data.
A tensor is produced by a producer op and can have multiple consumer ops.
There are three types of tensors: intermediate, variable and constant. Variable and constant tensors are initialised with data, while intermediate tensors are not.

* **Constant tensors** contain data that cannot change.

* **Variable tensors** contain data that is always live and hence is never freed. Typically model weights are kept on the IPU between runs and are therefore defined as variable tensors.

* **Intermediate tensors** are not initialised with data and are live from the time they are produced until their final consumer.

Operations
----------

An operation or op (:numref:`sec_operations`) represents an operation in the computational graph and can have input and output tensors.
