# Glossary

| **NOTE**: This glossary complements the [user guide](../user_guide/glossary.rst) glossary with components that are relevant for developers of PopART, but most likely not relevant to the end-user. |
| --- |

## Fragment

A `poplar` sequence.
A `subgraph` lowered to `poplar`.

## Graph

In PopART there is only one main-graph which is a directed acyclic graph (DAG) of `Ops`, connected to each other via `Tensors`.

### Subgraph

A `Graph` object which is not the main graph.
Called by a `CallOp` in the top level graph.
Not to be confused with a [virtual graph](#virtual-graph).
Also know as a "segment".

### Virtual graph

A graph created for a physical target.
Not to be confused with a [subgraph](#subgraph).
A view of the main graph.
Will add an annotation to the IR.