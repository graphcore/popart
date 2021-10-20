# Glossary

| **NOTE**: This glossary complements the [user guide](../user_guide/glossary.rst) glossary with components that are relevant for developers of PopART, but most likely not relevant to the end-user. |
| --- |

## Aliasing

An alias is a subset of the memory of a data structure (like a tensor) such that a change to the alias will change the subset in the original data structure
In other words:
The noun "alias", as in "t0 is an alias of t1" means "sharing the same location in memory".
I.e. "some or all of t0's memory location is the same as some or all of t1's memory location".

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

Do an operation on the memory location of the input instead of allocating new memory to the output.
This is the opposite of [outplacing](#outplacing)

## Outplacing

Allocate new memory to the output (i.e. do not overwrite the input memory).
This is the opposite of [inplacing](#inplacing)

## Outlining

A transformation that finds identical repeated clusters of `Ops` and extracts these clusters into [`subgraphs`](#subgraph), and replaces them with and `CallOps` that call these subgraphs.
This is akin to refactoring repeated lines of code into a function when using an imperative programming language.
The primary use of outlining is to reduce the amount of instruction memory that is used on an IPU.
