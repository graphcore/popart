# Glossary

| **NOTE**: This glossary complements the [user guide](../user_guide/glossary.rst) glossary with components that are relevant for developers of PopART, but most likely not relevant to the end-user. |
| --- |

## Aliasing

Declaration of a shared memory location between two tensors.
If `t0` is an alias of `t1` then some or all of `t0`'s memory location is the same as some or all of `t1`'s memory location.

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

## Inplacing

Inplacing is the process of reusing the memory used by an operation's input tensor for an output tensor of the operation. PopART uses inplacing to reduce the amount of variable memory required.
This is the opposite of [outplacing](#outplacing)

## Outplace

If an `Op` `isOutplace` then new memory will be allocated to the output tensor (i.e. it will not overwrite the input memory).
This is the opposite of [inplacing](#inplacing).

## Outlining

A transformation that finds identical repeated clusters of `Ops` and extracts these clusters into [`subgraphs`](#subgraph), and replaces them with and `CallOps` that call these subgraphs.
This is akin to refactoring repeated lines of code into a function when using an imperative programming language.
The primary use of outlining is to reduce the amount of instruction memory that is used on an IPU.
