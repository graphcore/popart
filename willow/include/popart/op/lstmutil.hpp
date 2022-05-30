// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LSTMUTIL_HPP
#define GUARD_NEURALNET_LSTMUTIL_HPP

#include <iostream>
#include <string>

namespace popart {

enum class ActivationFunction {
  Sigmoid = 0,
  Relu,
  Tanh,
  Gelu,
  Swish,
  Softmax,
  SoftmaxStable,
  SoftmaxScaled,
  N,
  Invalid
};

ActivationFunction fromString(const std::string &);

std::ostream &operator<<(std::ostream &, const ActivationFunction &);

} // namespace popart

#endif
