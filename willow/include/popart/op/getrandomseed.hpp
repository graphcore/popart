// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_GETRANDOMSEED_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_GETRANDOMSEED_HPP_

#include <memory>
#include <string>
#include <popart/op.hpp>
#include <popart/tensornames.hpp>

#include "popart/names.hpp"

namespace popart {
class AliasModel;
struct OperatorIdentifier;

class GetRandomSeedOp : public Op {
public:
  GetRandomSeedOp(const OperatorIdentifier &_opid,
                  const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;
  void setup() final;

  InIndex getSeedInIndex() const override { return 0; }
  static OutIndex getUpdatedSeedOutIndex() { return 0; }

  // Seeds are never batched
  int getOutBatchAxis(OutIndex) const override { return -1; }

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

  virtual void growAliasModel(AliasModel &m) const override {
    growAliasModelMulti(m);
  }
};
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_GETRANDOMSEED_HPP_
