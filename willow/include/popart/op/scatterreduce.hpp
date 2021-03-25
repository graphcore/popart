// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SCATTERREDUCE_HPP
#define GUARD_NEURALNET_SCATTERREDUCE_HPP

#include <popart/op.hpp>

namespace popart {

// TODO(T35695): when poplibs supports it, add Mul, Min, Max, etc
enum class ScatterReduction { Sum = 0 };

class ScatterReduceOp : public Op {
public:
  ScatterReduceOp(const OperatorIdentifier &_opid,
                  int64_t axis_,
                  int64_t axis_size_,
                  ScatterReduction reduction_,
                  const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  // Which axis to scatter reduce on.
  int64_t getAxis() const { return axis; }

  ScatterReduction getReduction() const { return reduction; }

  const Shape &getBackwardShape() const { return backward_shape; }

  static InIndex dataInIndex() { return 0; }
  static InIndex indicesInIndex() { return 1; }
  static OutIndex outIndex() { return 0; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  static std::string reductionToString(ScatterReduction reduction);
  static ScatterReduction reductionFromString(const std::string &reductionStr);

private:
  Shape backward_shape;
  int64_t axis;
  int64_t axis_size;
  ScatterReduction reduction;
};

class ScatterReduceGradOp : public Op {
public:
  ScatterReduceGradOp(const ScatterReduceOp &op);

  std::unique_ptr<Op> clone() const final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;

  int64_t getAxis() const { return axis; }

  ScatterReduction getReduction() const { return reduction; }

  static InIndex gradInIndex() { return 0; }
  static InIndex indicesInIndex() { return 1; }
  static OutIndex gradOutIndex() { return 0; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  Shape backward_shape;
  int64_t axis;
  ScatterReduction reduction;
};

} // namespace popart

#endif
