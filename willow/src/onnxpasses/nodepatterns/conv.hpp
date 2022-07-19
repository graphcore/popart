// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_SRC_ONNXPASSES_NODEPATTERNS_CONV_HPP_
#define POPART_WILLOW_SRC_ONNXPASSES_NODEPATTERNS_CONV_HPP_

#include <memory>
#include <onnxpasses/nodepattern.hpp>

#include "onnxpasses/onnxnames.hpp"

namespace popart {
namespace onnxpasses {
class PatternTarget;

// Replace:
//    [U1, ...] = MultiConv([X1, Y1, b1], ...)
//
// With:
//    [V1, ...] = MultiConv([X1, Y1], ...)
//    [U1, ...] = [addBias(V1, b1), ...]
//
// Where bias is an optional tensor input for each conv.
class MultiConvWithBias : public NodePattern {
public:
  MultiConvWithBias(std::shared_ptr<PatternTarget> t) : NodePattern(t) {}

private:
  bool go(const NodeProto &node) final;
};

// Replace Conv(X, Y, b) with addBias(Conv(X, Y), b)
class ConvWithBias : public NodePattern {
public:
  ConvWithBias(std::shared_ptr<PatternTarget> t) : NodePattern(t) {}

private:
  bool go(const NodeProto &node) final;
};

} // namespace onnxpasses
} // namespace popart

#endif // POPART_WILLOW_SRC_ONNXPASSES_NODEPATTERNS_CONV_HPP_
