#ifndef GUARD_NEURALNET_GRADIENTACCL_HPP
#define GUARD_NEURALNET_GRADIENTACCL_HPP

#include <popart/op.hpp>

namespace popart {

class GradientAcclOp : public Op {
public:
  GradientAcclOp(const OperatorIdentifier &opid, const Op::Settings &settings_);
  void setup() final;

  static InIndex getAcclInIndex() { return 0; }
  static InIndex getGradInIndex() { return 1; }

  static OutIndex getAcclOutIndex() { return 0; }
  // This Op modifies the input at index getAcclInIndex()
  view::Region modifies(InIndex) const final;

  view::Region aliases(InIndex) const final;

  float getSubgraphValue() const final { return 0.1f; }

  std::unique_ptr<Op> clone() const final;
};

class ResetAcclOp : public Op {
public:
  ResetAcclOp(const OperatorIdentifier &opid, const Op::Settings &settings_);
  void setup() final;

  static InIndex getAcclInIndex() { return 0; }

  static OutIndex getAcclOutIndex() { return 0; }

  // This Op modifies the input at index getAcclInIndex()
  view::Region modifies(InIndex) const final;
  view::Region aliases(InIndex) const final;

  float getSubgraphValue() const final { return 0.1f; }

  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif
