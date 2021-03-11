// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_AUTOMATICLOSSSCALE_HPP
#define GUARD_NEURALNET_AUTOMATICLOSSSCALE_HPP

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

// Before:
//
// lossGrad -- ... -- ConvWeightsGrad -- t0 -- ...
//  (*= ls)    ... -- Matmul -- t1 -- ...

// After:
//
// lossGrad -- ... -- ConvWeightsGrad --- t0 -- ...
//  (*= ls)                                |
//                                          - HistogramOp -- t0_stats
//             ... -- Matmul -- t1 -- ...                       |
//                               |                              |
//                                - HistogramOp -- t1_stats     |
//                                                      |       |
//                                                      |       |
//                                       ls ------ LossScaleUpdateOp
//                                   inverse_ls ---'     ||
//                                                       |'- ls_updated
//                                                       '-- inverse_ls_updated
namespace popart {

class AutomaticLossScale : public Transform {
public:
  static std::size_t id();

  AutomaticLossScale() : Transform() {}
  virtual ~AutomaticLossScale() override {}

  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "AutomaticLossScale"; }
};

} // namespace popart

#endif
