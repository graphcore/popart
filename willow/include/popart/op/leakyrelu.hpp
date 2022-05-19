// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LRELU_HPP
#define GUARD_NEURALNET_LRELU_HPP

#include <memory>
#include <tuple>
#include <vector>
#include <popart/op/elementwise.hpp>

#include "popart/op.hpp"

namespace popart {
class OpSerialiserBase;
struct OperatorIdentifier;

class LeakyReluOpBaseAttributes {
public:
  LeakyReluOpBaseAttributes(float _alpha) : alpha(_alpha) {}

  float getAlpha() const { return alpha; }

private:
  float alpha;
};

class LeakyReluOp : public ElementWiseUnaryOp,
                    public LeakyReluOpBaseAttributes {
public:
  LeakyReluOp(const OperatorIdentifier &_opid,
              float _alpha,
              const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;

  void appendAttributes(popart::OpSerialiserBase &os) const override;
  void appendOutlineAttributes(popart::OpSerialiserBase &os) const override;

  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;
};

class LeakyReluInplaceOp : public ElementWiseInplaceUnaryOp,
                           public LeakyReluOpBaseAttributes {
public:
  LeakyReluInplaceOp(const LeakyReluOp &);
  std::unique_ptr<Op> clone() const final;

  void appendAttributes(popart::OpSerialiserBase &os) const override;
  void appendOutlineAttributes(popart::OpSerialiserBase &os) const override;
};

class LeakyReluGradOp : public Op, public LeakyReluOpBaseAttributes {
public:
  LeakyReluGradOp(const LeakyReluOp &);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;
  std::unique_ptr<Op> clone() const final;

  void appendAttributes(popart::OpSerialiserBase &os) const override;
  void appendOutlineAttributes(popart::OpSerialiserBase &os) const override;
  // The index at which the output of the LeakyRelu
  // is an input to this LeakyReluGradOp
  static InIndex getLeakyReluInIndex() { return 1; }

  // Get the input index for the gradient of the output of the LeakyRelu, for
  // this LeakyReluGradOp.
  static InIndex getGradLeakyReluInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }
  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

} // namespace popart

#endif
