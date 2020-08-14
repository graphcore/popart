// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_RESIZE_HPP
#define GUARD_NEURALNET_RESIZE_HPP

#include <popart/op.hpp>

namespace popart {

enum class ResizeMode { Nearest, Linear, N };
std::string toString(const ResizeMode &);
std::ostream &operator<<(std::ostream &, const ResizeMode &);

class ResizeOp : public Op {
public:
  ResizeOp(const OperatorIdentifier &,
           const Op::Settings &,
           ResizeMode,
           const std::vector<float> &scales);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  ResizeMode getMode() const { return mode; }
  const std::vector<float> &getScales() const { return scales; }

private:
  const std::vector<float> scales;
  const ResizeMode mode;
};

class ResizeGradOp : public ResizeOp {
public:
  ResizeGradOp(const ResizeOp &);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
};

} // namespace popart

#endif
