// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_AUTOMATICLOSSSCALING_HPP_
#define POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_AUTOMATICLOSSSCALING_HPP_

#include <cstddef>
#include <map>
#include <string>
#include <popart/names.hpp>
#include <popart/transforms/transform.hpp>

// For context: why do we support loss scaling in our optimizers?
// We perform gradient descent to update Variable tensors stored in floating
// point format (either fp16 or fp32). These formats can express numbers within
// a particular range.
//
// It is the responsibility of the user to design a model that, in training,
// produces weight updates that are stable (i.e. they do not decay to zero,
// or grow exponentially).
//
// But in addition to this, without automatic loss scaling, the user must also
// ensure that operations in their model are producing tensor values that lie
// within the floating point range. If gradient tensors are consistently
// underflowing (rounded down to zero), or overflowing (clipped to the maximum
// expressible value of the number format), then they may observe unwanted
// training behaviour, such as convergence at a lower-than-expected accuracy.
// This problem is particularly common for fp16 gradients, where the dynamic
// range is much smaller.
//
// The aim of automatic loss scaling is to unburden the user of this
// responsiblity. Simply, the transform aims to adjust the loss scale during
// training, without any user input, in order to keep the gradient tensor values
// distributed evenly across the dynamic range of the floating point number
// format.
//
// It does this by collecting the statistics of the gradient tensors.
// HistogramOps are inserted into the graph to count the number of tensor
// elements in the lower and each of the upper and lower halves of the
// expressible range of the number format (after first taking the absolute
// values).
//
// Take the example 4-element, rank-1 fp16 tensor: {1e-3, -1.2e-4, -2, 1.4e5}.
// These elements are sorted into 'lower' and 'upper' bins, to output {3, 1}.
//
// Note that it is computationally inefficient to gather these statistics for
// every gradient tensor. The input and output of all view-changing ops, for,
// example would have exactly the same statistics. So we choose only a subset of
// grad ops for which we gather these statistics. These are described in the
// code as 'to-track' or 'tracked' tensors.

// These 'gradient statistics' are combined in the LossScaleUpdateOp to produce
// cumulative statistics. These are used to update the scalar loss scale tensor.
// Note that our optimizer implementations require separate scalars in the model
// for applying the loss scale (at the start of the backwards pass) and the
// inverse loss scale (applied just before the weight update). So the
// LossScaleUpdateOp takes both of these scalars as inputs, and outputs updated
// versions.

// In the simplest case, the transform is as follows.
//
// Before:
//
// lossGrad -- ... -- ConvWeightsGrad -- t0 -- ...
//  (*= ls)    ... -- Matmul -- t1 -- ...

// After:

// (Case 0, No graph replication or gradient accumulation)
//
// lossGrad -- ... -- ConvWeightsGrad --- t0 -- ...
//  (*= ls)                                |
//                                          - HistogramOp -- t0_stats
//            ... -- Matmul -- t1 -- ...                     |
//                              |                            |
//                               - HistogramOp -- t1_stats   |
//                                                    |     /
//                                                    SumOp
//                                                      |
//                                                 stats_summed
//                                                      |
//               ls_update_factor -----------> LossScaleUpdateOp
//                       |                              |
//                       |                    ls_update_factor_updated
//                       |  ls -.
//                       |--> MulOp -- final_ls
//                       |
//                       | inverse_ls -.
//                       '---------> DivOp -- final_inverse_ls

// (Case 1, With graph replication enabled)
//
//                     .. HistogramOp -- t0_stats -.
//                                                  \
//                     .. HistogramOp -- t1_stats -- SumOp
//                                                    |
//                                               stats_summed
//                                                    |
//                                                  CastOp
//                                                    |
//                                             stats_summed_fp32
//                                                    |
//                                           ReplicatedAllReduceOp
//                                                    |
//                                            stats_summed_reduced
//                                                    |
//               ls_update_factor -----------> LossScaleUpdateOp
//                       |                              |
//                       |                    ls_update_factor_updated
//                       |  ls -.
//                       |--> MulOp -- final_ls
//                       |
//                       | inverse_ls -.
//                       '---------> DivOp -- final_inverse_ls

// (Case 2, With gradient accumulation enabled)
//
// .. HistogramOp -- t0_stats -.
//                              \
// .. HistogramOp -- t1_stats -- SumOp
//                                |
//                           stats_summed
//                                |
//                              CastOp
//                                |
//                         stats_summed_fp32   stats_to_accl
//                                |           /    |
//                             AddRhsInplaceOp     - AccumulatorZeroOp (a.o.f)
//                                    |                         |
//                               stats_accld             stats_to_accl_reset
//                                    |
//  ls_update_factor -----------> LossScaleUpdateOp (a.o.f)
//          |                              |
//          |                    ls_update_factor_updated
//          |  ls -.
//          |--> MulOp -- final_ls
//          |
//          | inverse_ls -.
//          '---------> DivOp -- final_inverse_ls
//
// (where a.o.f means is assigned to the
//  ExecutionContext::AccumulateOuterFragment)

// (Case 3, With both graph replication and gradient accumulation enabled)
//
// .. HistogramOp -- t0_stats -.
//                              \
// .. HistogramOp -- t1_stats -- SumOp
//                                |
//                           stats_summed
//                                |
//                              CastOp
//                                |
//                         stats_summed_fp32   stats_to_accl
//                                |           /    |
//                             AddRhsInplaceOp     - AccumulatorZeroOp (a.o.f)
//                                    |                         |
//                               stats_accld             stats_to_accl_reset
//                                    |
//                        ReplicatedAllReduceOp (a.o.f)
//                                    |
//                            stats_accld_reduced
//                                    |
//  ls_update_factor -----------> LossScaleUpdateOp (a.o.f)
//          |                              |
//          |                    ls_update_factor_updated
//          |  ls -.
//          |--> MulOp -- final_ls
//          |
//          | inverse_ls -.
//          '---------> DivOp -- final_inverse_ls
//
// (where a.o.f means is assigned to the
//  ExecutionContext::AccumulateOuterFragment)
//
// executeOpNTimesEveryMTimes:
//
// A function that transforms the graph in a way changes the frequency
// of execution of an operation.
// The user provides two positive integers - N and M - that control
// the frequency of execution of the operation.
// Take the example where N=2, M=6.
// The execution pattern of the op will be:
// Execution is:
// Yes, Yes, No, No, No, No, Yes, Yes, No, No, No, No, Yes, Yes,
// No, No, No, No

// How it works:
// It builds a counter tensor for the input op.
// The counter is incremented by one for each normal execution of the op.
// There is a simple logic provided by modulo, less operations and N, M to set
// the execution flag for if op.
// It creates an if op with its two branches. The op is moved to one branch
// which is for executing the op.
// The second branch is an "empty subgraph" for not executing the op.
// For the "empty subgraph" it is possible to connect inputs and outputs
// via nop. And it is possible to set up default values of outputs
// by creating scalars and expanding them to the outputs.
// Those are set up by two maps identityInputToOutputIndiciesMapping and
// outputIndiciesAndValues.
//
// Before using executeOpNTimesEveryMTimes():
//  t0   t1 ...
//   \  /
//    OpA
//   /  \
// t_k  t_k+1 ...
//
// After:
//   counter
//     |
//     v
//   IncrementModInplaceOp(1, M)
//     |
//    flag_inc
//     |
//    LessOp(N)   t0   t1 ...
//     |         \  /
//    flag-------IfOp(Sg0, Sg1)
//               /  \
//             t_k   t_k+1 ...
//
// Where Sg0:
//     t0  t1 ...
//     |   |
//     v   v
//  ----\--|------
// |     \ |      |
// |      OpA     |
// |     / |      |
//  ----/--|------
//     v   v
//    /    |
//  t_k   t_k+1 ...
//
// Where Sg1:
//     t0  t1 ...
//     |   |
//     v   v
//  ---|---|--------------------
// |       |                    |
// |      Nop     0->Expand     |
// |     /             |        |
//  ----/--|-----------|--------
//     v   v           v
//    /    |           |
//  t_k   t_k+1 ...  t_k+7 ...
//
// t_in, t_out, Nop, Expand connections are specified by user.
namespace popart {
class Op;
class AliasModel;
class Graph;

class AutomaticLossScale : public Transform {
public:
  static std::size_t id();

  /**
   * When applied to an op it will be effectively executed
   * n times every m times.
   * It returns a pointer to an IfOp which either calls an 'empty' subgraph,
   * or calls a subgraph containing the op passed as the argument.
   * The 'empty' subgraph is meant to be low intensity compute.
   * It is possible to connect inputs and outputs via nop operations
   * and set up default values of outputs in the 'empty' subgraph.
   * \param op Operator whose execution frequency is modified.
   * \param n Execute the op n times every m times.
   * \param m Execute the op n times every m times.
   * \param identityInputToOutputIndiciesMapping Specifies the connections
   *  of inputs to outputs via nop operations in the 'empty' subgraph.
   *  Each pair must have the same shape and type.
   * \param outputIndiciesAndValues Map of pairs of output indices and values.
   * Note: inplacing and aliasing of inputs are not supported.
   * If the op inplace-modifies or aliases an input, in the transformed graph
   * after this method is called, this will not longer be the case.
   */
  static Op *executeOpNTimesEveryMTimes(
      Op *op,
      unsigned n,
      unsigned m,
      const std::map<InIndex, OutIndex> &identityInputToOutputIndiciesMapping,
      const std::map<OutIndex, float> &outputIndiciesAndValues,
      AliasModel &aliasMode);

  AutomaticLossScale() : Transform() {}
  virtual ~AutomaticLossScale() override {}

  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "AutomaticLossScale"; }
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_AUTOMATICLOSSSCALING_HPP_
