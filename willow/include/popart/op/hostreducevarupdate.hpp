// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_HOSTREDUCEVARUPDATE_HPP
#define GUARD_NEURALNET_HOSTREDUCEVARUPDATE_HPP

#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/sgd0varupdate.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/optimizervalue.hpp>

namespace popart {

// Copy gradients to the host. Does not produce any output
class GradCopyToHostOp : public Op {
public:
  GradCopyToHostOp(const Op::Settings &);
  std::unique_ptr<Op> clone() const final;

  // Gradient index
  static InIndex getInIndex() { return 0; }

  void setup() final {}

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  void appendAttributes(OpSerialiserBase &) const final;

  bool hasSideEffect() const override { return true; }
};

// Copy gradients to the device. Output is the streamed gradient
class GradCopyFromHostOp : public Op {
public:
  GradCopyFromHostOp(const Op::Settings &);
  std::unique_ptr<Op> clone() const final;

  // Gradient index
  static InIndex getInIndex() { return 0; }

  static OutIndex getOutIndex() { return 0; }

  void setup() final;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  void appendAttributes(OpSerialiserBase &) const final;
};

// Copy variables from the host
class HostSGD0VarUpdate : public SGD0VarUpdateOpBase {
public:
  HostSGD0VarUpdate(OptimizerValue initialScaledLearningRate,
                    OptimizerValue initialWeightDecayScaleFactor,
                    const Op::Settings &);
  std::unique_ptr<Op> clone() const final;

  float getSubgraphValue() const final { return getHighSubgraphValue(); }
};

} // namespace popart

#endif
