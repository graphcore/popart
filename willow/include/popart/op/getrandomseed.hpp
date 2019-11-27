#ifndef GUARD_NEURALNET_GETRANDOMSEED_HPP
#define GUARD_NEURALNET_GETRANDOMSEED_HPP

#include <popart/op.hpp>
#include <popart/tensornames.hpp>

namespace popart {

class GetRandomSeedOp : public Op {
public:
  GetRandomSeedOp(const OperatorIdentifier &_opid,
                  const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;
  void setup() final;

  InIndex getSeedInIndex() const override { return 0; }
  static OutIndex getUpdatedSeedOutIndex() { return 0; }

  static TensorId getStreamedSeedTensorId() {
    return reservedRandomSeedPrefix() + std::string("fromHost");
  }
  static TensorId getUpdatedSeedTensorId() {
    return reservedRandomSeedPrefix() + std::string("updated");
  }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  bool isOutlineable() const final { return false; }

  // This updated seed aliases and in-place modifies the input seed
  view::Regions aliases(InIndex, OutIndex) const final;
  view::Regions modifies(InIndex) const final;
};
} // namespace popart

#endif
