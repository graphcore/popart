// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_GROUPNORM_HPP
#define GUARD_NEURALNET_GROUPNORM_HPP

#include <cstdint>
#include <map>
#include <memory>
#include <vector>
#include <popart/op.hpp>

#include "popart/names.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
class OpSerialiserBase;
struct OperatorIdentifier;

class GroupNormOp : public Op {
public:
  GroupNormOp(const OperatorIdentifier &opid_,
              int64_t num_groups_,
              float epsilon_,
              const Op::Settings &settings);

  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  // Input's
  static InIndex getXInIndex() { return 0; }
  static InIndex getScaleInIndex() { return 1; }
  static InIndex getBInIndex() { return 2; }

  // Ouput's
  static OutIndex getYOutIndex() { return 0; }
  static OutIndex getMeanOutIndex() { return 1; }
  static OutIndex getInvStdDevOutIndex() { return 2; }

  // Attributes
  float getEpsilon() const { return epsilon; }
  int64_t getNumGroups() const { return num_groups; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  bool isNorm() const override { return true; }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  bool canShard() const override { return true; }

  bool canBeReplacedByIdentity() const final;

private:
  int64_t num_groups;
  float epsilon;
};

class GroupNormGradOp : public Op {
public:
  GroupNormGradOp(const GroupNormOp &);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;
  std::unique_ptr<Op> clone() const final;

  static InIndex getXInIndex() { return 0; }
  static InIndex getScaleInIndex() { return 1; }
  static InIndex getMeanInIndex() { return 2; }
  static InIndex getInvStdDevInIndex() { return 3; }
  static InIndex getYGradInIndex() { return 4; }

  static OutIndex getXGradOutIndex() { return 0; }
  static OutIndex getScaleOutIndex() { return 1; }
  static OutIndex getBOutIndex() { return 2; }

  float getEpsilon() const { return epsilon; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  bool canShard() const override { return true; }

private:
  float epsilon;
  TensorInfo fwdInInfo, fwdScaleInInfo, fwdBInInfo;
};

} // namespace popart

#endif
