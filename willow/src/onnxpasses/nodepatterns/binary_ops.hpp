// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ONNXTOONNX_BINARY_OPS_HPP
#define GUARD_NEURALNET_ONNXTOONNX_BINARY_OPS_HPP

#include <onnxpasses/nodepattern.hpp>

namespace popart {
namespace onnxpasses {

// Replace 'ai.graphcore.remainder' with:
//   ai.graphcore.fmod(arg1 + ai.graphcore.fmod(arg0, arg1), arg1)
class Remainder : public NodePattern {
public:
  explicit Remainder(std::shared_ptr<PatternTarget> t)
      : NodePattern(std::move(t)) {}

private:
  bool go(const NodeProto &node) override final;
};

} // namespace onnxpasses
} // namespace popart

#endif /* !GUARD_NEURALNET_ONNXTOONNX_BINARY_OPS_HPP */
