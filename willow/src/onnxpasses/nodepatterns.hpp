// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ONNXTOONNX_NODEPATTERNS_HPP
#define GUARD_NEURALNET_ONNXTOONNX_NODEPATTERNS_HPP

#include <onnxpasses/nodepattern.hpp>

namespace popart {
namespace onnxpasses {

// Replace Conv(X, Y, b) with addBias(Conv(X, Y), b)
class ConvWithBias : public NodePattern {
public:
  ConvWithBias(Graph &g, Suffixer &s) : NodePattern(g, s) {}

private:
  bool go(const Node &node) final;
};

// Replace tan(x) with div(sin(x),cos(x))
class Tan : public NodePattern {
public:
  Tan(Graph &g, Suffixer &s) : NodePattern(g, s) {}

private:
  bool go(const Node &node) final;
};

class Asinh : public NodePattern {
public:
  Asinh(Graph &g, Suffixer &s) : NodePattern(g, s) {}

private:
  bool go(const Node &node) final;
};

// acosh(x) = ln(x + sqrt(x^2 - 1) ); Defined for [1, +inf)
class Acosh : public NodePattern {
public:
  Acosh(Graph &g, Suffixer &s) : NodePattern(g, s) {}

private:
  bool go(const Node &node) final;
};

// acos(x) = pi / 2 - asin(x)
class Acos : public NodePattern {
public:
  Acos(Graph &g, Suffixer &s) : NodePattern(g, s) {}

private:
  bool go(const Node &node) final;
};

// atanh(x) = 1/2 ln( (1 + x) / (1 - x) )
class Atanh : public NodePattern {
public:
  Atanh(Graph &g, Suffixer &s) : NodePattern(g, s) {}

private:
  bool go(const Node &node) final;
};

} // namespace onnxpasses
} // namespace popart

#endif
