// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_ONEHOT_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_ONEHOT_HPP_

#include <cstdint>
#include <map>
#include <memory>
#include <vector>
#include <popart/op.hpp>

#include "popart/names.hpp"

namespace popart {
class OpSerialiserBase;
struct OperatorIdentifier;

// This Op is based on the ONNX Operator described at
// github.com/onnx/onnx/blob/master/docs/Operators.md#Onehot
// but it is slightly different: this Op is static w.r.t. depth
class OnehotOp : public Op {
public:
  OnehotOp(const OperatorIdentifier &_opid,
           int64_t axis_,
           const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  void setup() override;

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  // The depth input is not connected to Onehot as an input
  // but rather is read in the connectInTensor method and used
  // to set the onehotAxisDim member variable

  static InIndex getIndicesInIndex() { return 0; }
  static InIndex getValuesInIndex() { return 2; }

  static OutIndex getOutIndex() { return 0; }

  virtual void connectInTensor(InIndex inIndex, TensorId tenId) override;

  int64_t getAxis() const { return axis; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  int64_t axis;
  int64_t onehotAxisDim;
};

class OnehotGradOp : public Op {
public:
  OnehotGradOp(const OnehotOp &fwdOp_);
  std::unique_ptr<Op> clone() const final;

  void setup() override;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  static InIndex getGradInIndex() { return 0; }
  static InIndex getIndicesInIndex() { return 1; }

  static OutIndex getOutIndex() { return 0; }

  const Shape &getOutputShape() const { return outputShape; }

  int64_t getAxis() const { return axis; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  int64_t axis;
  Shape outputShape;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_ONEHOT_HPP_
