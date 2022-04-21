// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_DROPOUTBASE_HPP
#define GUARD_NEURALNET_DROPOUTBASE_HPP

#include <popart/op.hpp>
#include <popart/op/randombase.hpp>

namespace popart {

// Forward declare for use in static function signature
class OpCreatorInfo;

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

#endif
