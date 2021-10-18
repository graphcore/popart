# Pipelining

For an introduction of pipelining, see [pipelining in TensorFlow](https://docs.graphcore.ai/projects/tf-model-parallelism/en/latest/pipelining.html).

See [pipelining in the popART glossary](https://docs.graphcore.ai/projects/popart-user-guide/en/latest/glossary.html#pipelining) for explanation of technical terms.

## Transformation of a pipeline stage

Consider a graph, sharded over 3 virtual graphs, and split into 3 pipeline stages

```text
in -> ps0 -> copy -> ps1 -> copy -> ps2 -> out
```

Where

```text
in         denotes the host to device copying
ps<number> denotes the pipeline stage
copy       denotes the inter IPU copying
out        denotes the device to host copying
```

The pipelined graph with `N` `PipelineCycles` then looks like:

```text
P = number of pipeline stages
G = number of gradient accumulation OR batches per step
N = number of cycles = 2 * (P - 1) + (G - (P - 1))

{     fill phase (P-1 cycles)    } {----------- main phase -----------} { flush phase (P-1 cycles) }
 Pipeline        Pipeline               Main Pipeline Cycle              Pipeline       Pipeline
 Cycle 0         Cycle 1            (Repeat G-(P-1)=N-2(P-1) times)      Cycle N-2      Cycle N-1

                                      .--- < loop carried < ----.
                                      |                         |
in -> ps0 -> copy -> ps1 -> copy -> loop_in -> ps2 -> out       |                                     } parallel
               in -> ps0 -> copy -> loop_in -> ps1 -> copy -> loop_out -> ps2 -> out                  } execution
                                         in -> ps0 -> copy -> loop_out -> ps1 -> copy -> ps2 -> out   } slots
```

## Assemble from fragments

This section describes how pipeline is constructed during Ir-lowering.

The way we assemble the full pipeline program from program `fragment`s is based on two ideas:

1. Constraints are imposed on the order of `fragment`s by `poplar` lowering optimisations to guarantee parallel execution over IPUs.
   A `poplar::program` is constructed serially, like:

   ```c++
   poplar::Program::Sequence seq;
   seq.add(fragment0);
   seq.add(fragment1);
   ...
   seq.add(fragmentN);
   ```

   But a successfully pipelined model will have maximally parallelised
   execution over IPUs. For `fragment`s 0 to N to be parallelisable, they
   must:

     - run on different IPUs to one another
     - not entail a global exchange

   In `PopART` we enforce this by splitting operations assigned to each Pipeline Stage into four `fragment`s:
   `ToDeviceStream` (`D`), `Main` (`M`), `ToHostStream` (`H`), and `IpuCopy` (`C`). 
   The program for a Pipeline Cycle is assembled by:

     - Adding to the program `D` `fragment`s for all Pipeline Stages participating in the Pipeline Cycle.
     - Followed by `M` `fragment`s for all Pipeline Stages participating in the Pipeline Cycle.
     - Followed by `H` `fragment`s for all Pipeline Stages participating in the Pipeline Cycle.
     - Followed by `C` `fragment`s for all Pipeline Stages (see (2) for an explanation)

   The full pipeline program is then assembled from the programs for each Pipeline Cycle.

2. Not all Pipeline Stages execute in every Pipeline Cycle.
   The full program starts with a 'fill phase' and ends with a 'flush phase', each consisting of Pipeline Cycles in which some Pipeline Stages do not participate.
   The exception to this is the inter-IPU copy `fragment`. In order to get Poplar to run these in parallel over IPUs inside the 'main' Pipeline Cycle, they must run every Pipeline Cycle for all Pipeline Stages.

To illustrate these two ideas, consider the model with three layers in the forward pass:

```text
StreamFromHost
   |
   v
   A->A' IPU0
   |  ^
   v  |
   B->B' IPU1
   |  ^
   v  |
   C->C' IPU2     (where X' is the grad layer of X)
   |
   v
StreamToHost
```

A simple layer-to-pipeline-stage mapping (either set by the user or
inferred automatically based on `VirtualGraph` mapping) could be:

| Pipline Stage |  Layers |
|---------------|---------|
| 0             |  {A}    |
| 1             |  {B}    |
| 2             |  {C}    |

After auto-grad is applied, the complete graph will then have the mapping:

| Pipline Stage |  Layers  |
|---------------|----------|
| 0             |  {A}     |
| 1             |  {B}     |
| 2             |  {C, C'} |
| 3             |  {B'}    |
| 4             |  {A'}    |

Note that in order to satisfy the requirement that 

> operations on a Pipeline Stage have no dependencies on other Pipeline Stages

layers that have dependents on other Pipeline Stages on the same IPU are augmented with `Stash` operations in the IR that copy their activations to a FILO buffer, or stash.
Also, layers that depend on other Pipeline Stages on the same IPU are augmented with `Restore` operations that restore their inputs from these stashes. The scheduling of these new operations are handled by the IR scheduler.

A pipeline with the minimum number of steps for 3 IPUs looks as follows:

```text
Pipeline Cycle -->

     <-------------- fill --------------> <- main > <-------- flush --------->
PS0: D0.M0.| D1.M1.| D2.M2   .| D3.M3   .|D4.M4   .|       |       |    |    |
PS1:       |    M0.|    M1   .|    M2   .|   M3   .| M4   .|       |    |    |
PS2:       C       C    M0.H0.C    M1.H1.C   M2.H2.C M3.H3.C M4.H4.C    C    C
PS3:       |       |          |    M0   .|   M1   .| M2   .| M3   .| M4.|    |
PS4:       |       |          |          |   M0   .| M1   .| M2   .| M3.| M4.|
```

Program `fragment` key:

- `D` - `ToDeviceStream`
- `M` - `Main`
- `H` - `ToHostStream`
- `C` - `IpuCopy`

`D<i>` means `fragment` `D` executes on [micro-batch](https://docs.graphcore.ai/projects/popart-user-guide/en/latest/glossary.html#batch-size) `i`, etc.

We can see from this diagram how the full pipeline program is assembled - starting in the top-left corner, serialise the 2D 'schedule' by reading down the columns:

```cpp
poplar::Program::Sequence seq;  // the full pipeline program
seq.add(ps0[D]);
seq.add(ps0[M]);
seq.add(C);
seq.add(ps0[D]);
seq.add(ps0[M]);
seq.add(ps1[M]);
seq.add(C);  // ... etc.
```

 We can also look at the full program from the perspective of IPU utilization, considering that Pipeline Stages on the same IPU must execute serially:

```
      <-------------- fill ---------------> <--- main ---> <--------------- flush --------------->
IPU0: PS0(0), PS0(1), PS0(2), PS0(3)       , PS0(4).PS4(0), PS4(1)        , PS4(2), PS4(3), PS4(4)
IPU1:         PS1(0), PS1(1), PS1(2).PS3(0), PS1(3).PS3(1), PS1(4).PS3(2) , PS3(3), PS3(4)
IPU2:                 PS2(0), PS2(1)       , PS2(2)       , PS2(3)        , PS2(4),
```

The holes in this second diagram represent idle-time for an IPU.
Maximizing utilization is therefore a case of

- Maximizing the proportion of 'main' Pipeline Cycles, achieved by having as large a [batches per step](https://docs.graphcore.ai/projects/popart-user-guide/en/latest/glossary.html#batches-per-step) (or gradient accumulation factor, if gradient accumulation is enabled).
- Optimally balancing the cycles required on each IPU in the main Pipeline Cycle
