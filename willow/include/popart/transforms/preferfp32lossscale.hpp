// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_PREFER_FP32_LOSS_SCALE_HPP
#define GUARD_NEURALNET_PREFER_FP32_LOSS_SCALE_HPP

#include <popart/transforms/transform.hpp>

namespace popart {

// Some background context:
// In PopART's autograd transform, if the TrainingSession is constructed with
// a loss TensorId corresponding to an fp16 tensor, and an optimizer with loss
// scaling, then an fp16 loss scale tensor is created in the the IR, and passed
// as an input to the loss grad op.
//
// Motivation for this pattern:
// We want to be able to support loss scale values > max(fp16) even in the case
// when the loss gradient op prodcues an fp16 output.
//
// The pattern:
// Match on ops that produce an fp16 gradient, but can take in an fp32 loss
// scale. Let's call these 'mixed precision loss grad ops', or MPLGOs.
// Replace cases (really only ever 1) of:
//    lossScale_fp16 -> MPLGO -> grad_fp16
// with:
//    lossScale_fp32 -> MPLGO -> grad_fp16
class PreferFp32LossScale : public Transform {
public:
  static std::size_t id();

  PreferFp32LossScale() : Transform() {}
  virtual ~PreferFp32LossScale() override {}

  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "PreferFp32LossScale"; }
};

} // namespace popart

#endif
