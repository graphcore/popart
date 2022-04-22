// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ONNXTOONNX_TRIG_HPP
#define GUARD_NEURALNET_ONNXTOONNX_TRIG_HPP

#include <memory>
#include <onnxpasses/nodepattern.hpp>

#include "onnxpasses/onnxnames.hpp"

namespace popart {
namespace onnxpasses {
class PatternTarget;

// Replace tan(x) with div(sin(x),cos(x))
class Tan : public NodePattern {
public:
  Tan(std::shared_ptr<PatternTarget> t) : NodePattern(t) {}

private:
  bool go(const NodeProto &node) final;
};

class Asinh : public NodePattern {
public:
  Asinh(std::shared_ptr<PatternTarget> t) : NodePattern(t) {}

private:
  bool go(const NodeProto &node) final;
};

// acosh(x) = ln(x + sqrt(x^2 - 1) ); Defined for [1, +inf)
class Acosh : public NodePattern {
public:
  Acosh(std::shared_ptr<PatternTarget> t) : NodePattern(t) {}

private:
  bool go(const NodeProto &node) final;
};

// acos(x) = pi / 2 - asin(x)
class Acos : public NodePattern {
public:
  Acos(std::shared_ptr<PatternTarget> t) : NodePattern(t) {}

private:
  bool go(const NodeProto &node) final;
};

// atanh(x) = 1/2 ln( (1 + x) / (1 - x) )
class Atanh : public NodePattern {
public:
  Atanh(std::shared_ptr<PatternTarget> t) : NodePattern(t) {}

private:
  bool go(const NodeProto &node) final;
};

} // namespace onnxpasses
} // namespace popart

#endif
