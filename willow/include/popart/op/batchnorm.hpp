// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_BATCHNORM_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_BATCHNORM_HPP_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <popart/op.hpp>

#include "popart/names.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
class OpSerialiserBase;
struct OperatorIdentifier;

class BatchNormOp : public Op {
public:
  BatchNormOp(const OperatorIdentifier &_opid,
              float _epsilon,
              float _momentum,
              int64_t _spatial,
              bool _unbiased_variance,
              const Op::Settings &settings);

  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  // Inputs
  static InIndex getXInIndex() { return 0; }
  static InIndex getScaleInIndex() { return 1; }
  static InIndex getBInIndex() { return 2; }
  static InIndex getMeanInIndex() { return 3; }
  static InIndex getVarInIndex() { return 4; }

  // Ouputs
  static OutIndex getYOutIndex() { return 0; }
  static OutIndex getMeanOutIndex() { return 1; }
  static OutIndex getVarOutIndex() { return 2; }
  static OutIndex getSavedMeanOutIndex() { return 3; }
  static OutIndex getSavedVarOutIndex() { return 4; }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  // Attributes
  float getEpsilon() const { return epsilon; }
  float getMomentum() const { return momentum; }
  int64_t getSpatial() const { return spatial; }
  bool useUnbiasedVariance() const { return unbiased_variance; }

  bool isTraining() const {
    (void)isTest;
    return training;
  }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  bool isNorm() const override { return true; }

private:
  void validateInput(const TensorInfo &inputInfo, const std::string &inputName);

  bool training = false;
  bool isTest;
  float epsilon;
  float momentum;
  int64_t spatial;
  bool unbiased_variance;
};

class BatchNormGradOp : public Op {
public:
  BatchNormGradOp(const BatchNormOp &);
  std::unique_ptr<Op> clone() const final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;

  static InIndex getXInIndex() { return 0; }
  static InIndex getScaleInIndex() { return 1; }
  static InIndex getMeanInIndex() { return 2; }
  static InIndex getVarInIndex() { return 3; }
  static InIndex getYGradInIndex() { return 4; }

  static OutIndex getXOutIndex() { return 0; }
  static OutIndex getScaleOutIndex() { return 1; }
  static OutIndex getBOutIndex() { return 2; }

  float getEpsilon() const { return epsilon; }
  int64_t getSpatial() const { return spatial; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

private:
  float epsilon;
  int64_t spatial;
  TensorInfo fwdInInfo, fwdScaleInInfo, fwdBInInfo;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_BATCHNORM_HPP_
