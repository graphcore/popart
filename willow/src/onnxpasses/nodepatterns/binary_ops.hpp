// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ONNXTOONNX_BINARY_OPS_HPP
#define GUARD_NEURALNET_ONNXTOONNX_BINARY_OPS_HPP

#include <memory>
#include <onnxpasses/nodepattern.hpp>
#include <utility>

#include "onnxpasses/onnxnames.hpp"

namespace popart {
namespace onnxpasses {
class PatternTarget;

// Replace 'ai.graphcore.remainder' with:
//   ai.graphcore.fmod(arg1 + ai.graphcore.fmod(arg0, arg1), arg1)
class Remainder : public NodePattern {
public:
  explicit Remainder(std::shared_ptr<PatternTarget> t)
      : NodePattern(std::move(t)) {}

private:
  bool go(const NodeProto &node) override;

protected:
  virtual void remainderToFmod(const NodeProto &node);
};

// If the "fmod" attribute is 1, then ai.onnx.mod gets replaced with
// ai.graphcore.Fmod. Otherwise, it's replaced with the same expression that is
// used in ai.graphcore.Remainder.
class Mod : public Remainder {
public:
  explicit Mod(std::shared_ptr<PatternTarget> t) : Remainder(std::move(t)) {}

private:
  bool go(const NodeProto &node) override final;
};

} // namespace onnxpasses
} // namespace popart

#endif /* !GUARD_NEURALNET_ONNXTOONNX_BINARY_OPS_HPP */
