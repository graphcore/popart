// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SHAPEDDROPOUT_HPP
#define GUARD_NEURALNET_SHAPEDDROPOUT_HPP

#include <popart/op/dropoutbase.hpp>

namespace popart {

class ShapedDropoutOp : public DropoutBaseOp {
public:
  ShapedDropoutOp(const OperatorIdentifier &_opid,
                  float ratio_,
                  const std::vector<int64_t> &shape_,
                  const Op::Settings &settings_,
                  RandomSeedPlaceholder placeholder_ = RandomSeedPlaceholder());

  const std::vector<int64_t> &getShape() const { return shape; }

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  void appendOutlineAttributes(OpSerialiserBase &) const final;

private:
  std::vector<int64_t> shape;
};

class ShapedDropoutGradOp : public ShapedDropoutOp {
public:
  ShapedDropoutGradOp(const ShapedDropoutOp &fwdOp);

  std::unique_ptr<Op> clone() const final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  static InIndex getGradInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }
};

} // namespace popart

#endif
