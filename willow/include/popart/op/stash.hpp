#ifndef GUARD_NEURALNET_STASH_HPP
#define GUARD_NEURALNET_STASH_HPP

#include <popart/op.hpp>

namespace popart {

class StashOp : public Op {
public:
  StashOp(const OperatorIdentifier &, const Op::Settings &);

  std::unique_ptr<Op> clone() const override;
  void setup() final;

  static InIndex getInIndex() { return 0; }
  static InIndex getOutIndex() { return 0; }

  int64_t getStashSize();
  TensorId getStashedTensorId() const;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

} // namespace popart

#endif
