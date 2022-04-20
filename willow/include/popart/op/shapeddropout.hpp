// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SHAPEDDROPOUT_HPP
#define GUARD_NEURALNET_SHAPEDDROPOUT_HPP

#include <cstdint>
#include <map>
#include <memory>
#include <vector>
#include <popart/op/dropoutbase.hpp>

#include "popart/names.hpp"
#include "popart/op.hpp"

namespace popart {
class OpSerialiserBase;
struct OperatorIdentifier;

class ShapedDropoutOp : public DropoutBaseOp {
public:
  ShapedDropoutOp(const OperatorIdentifier &_opid,
                  float ratio_,
                  const Shape &shape_,
                  const Op::Settings &settings_);

  const std::vector<int64_t> &getShape() const { return shape; }

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() override;
  void setup() override;

  void appendOutlineAttributes(OpSerialiserBase &) const override;

private:
  std::vector<int64_t> shape;
};

class ShapedDropoutGradOp : public ShapedDropoutOp {
public:
  ShapedDropoutGradOp(const ShapedDropoutOp &fwdOp);

  std::unique_ptr<Op> clone() const override;
  const std::vector<GradInOutMapper> &gradInputInfo() const override;
  const std::map<int, int> &gradOutToNonGradIn() const override;

  static InIndex getGradInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }
};

} // namespace popart

#endif
