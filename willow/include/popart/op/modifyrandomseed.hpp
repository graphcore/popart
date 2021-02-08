// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_MODIFYRANDOMSEED_HPP
#define GUARD_NEURALNET_MODIFYRANDOMSEED_HPP

#include <popart/op.hpp>
#include <popart/tensornames.hpp>

namespace popart {

class ModifyRandomSeedOp : public Op {
public:
  ModifyRandomSeedOp(const OperatorIdentifier &_opid,
                     const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;
  void setup() final;

  InIndex getSeedInIndex() const override { return 0; }
  static InIndex getSeedModifierInIndex() { return 1; }
  static OutIndex getModifiedSeedOutIndex() { return 0; }

  static TensorId getSeedModifierTensorId(const uint32_t modifier) {
    return reservedSeedModifierPrefix() + std::to_string(modifier);
  }
  static TensorId getModifiedSeedTensorId(const uint32_t modifier) {
    return reservedRandomSeedPrefix() + std::string("modified") +
           std::to_string(modifier);
  }

  // No batch dimensions in input/output tensors.
  int getOutBatchAxis(OutIndex) const override { return -1; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  bool isOutlineable() const final { return true; }
};
} // namespace popart

#endif
