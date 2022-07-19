// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_DROPOUTBASE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_DROPOUTBASE_HPP_

#include <memory>
#include <popart/op.hpp>
#include <popart/op/randombase.hpp>

#include "popart/names.hpp"

namespace popart {

// Forward declare for use in static function signature
class OpCreatorInfo;
struct OperatorIdentifier;

// Base class for dropout ops
class DropoutBaseOp : public RandomBaseOp {
public:
  DropoutBaseOp(const OperatorIdentifier &_opid,
                float ratio_,
                const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const override;

  // Inputs
  static InIndex getInIndex() { return 0; }

  // Outputs
  static OutIndex getOutIndex() { return 0; }

  bool canBeReplacedByIdentity() const override;

  float getRatio() const { return ratio; }
  void setRatio(float r) { ratio = r; }

  InIndex getSeedInIndex() const override { return 1; }

  bool canShard() const override { return true; }

  void configureShardedOp(Op *const shardedOp,
                          const Settings *const settings_) const override;

  static float validateRatioAttribute(const OpCreatorInfo &info);

private:
  float ratio;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_DROPOUTBASE_HPP_
