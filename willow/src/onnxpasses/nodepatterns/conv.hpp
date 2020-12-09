// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ONNXTOONNX_MULTICONVWITHBIAS_HPP
#define GUARD_NEURALNET_ONNXTOONNX_MULTICONVWITHBIAS_HPP

#include <onnxpasses/nodepattern.hpp>

namespace popart {
namespace onnxpasses {

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

#endif
