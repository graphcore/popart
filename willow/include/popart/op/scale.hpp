// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_SCALE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_SCALE_HPP_

#include <map>
#include <memory>
#include <tuple>
#include <vector>
#include <popart/op/elementwise.hpp>

#include "popart/names.hpp"
#include "popart/op.hpp"

namespace popart {

class OpSerialiserBase;
struct OperatorIdentifier;

// y = scale_factor * x
class ScaleOp : public ElementWiseUnaryOp {
public:
  ScaleOp(const OperatorIdentifier &_opid,
          float scale_,
          const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  void setScaleFactor(float value) { scale_factor = value; }
  float getScaleFactor() const;
  void appendOutlineAttributes(OpSerialiserBase &) const override;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;
  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;
  bool canBeReplacedByIdentity() const override;

private:
  float scale_factor;
};

class ScaleInplaceOp : public ElementWiseInplaceUnaryOp {
public:
  ScaleInplaceOp(const ScaleOp &);
  ScaleInplaceOp(const OperatorIdentifier &_opid,
                 float scale_,
                 const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const final;

  // TODO T6801 : don't repeat scale_factor
  float getScaleFactor() const;
  void appendOutlineAttributes(OpSerialiserBase &) const override;

private:
  float scale_factor;
};

class ScaleGradOp : public ScaleOp {
public:
  ScaleGradOp(const ScaleOp &fwdOp);
  std::unique_ptr<Op> clone() const final;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_SCALE_HPP_
