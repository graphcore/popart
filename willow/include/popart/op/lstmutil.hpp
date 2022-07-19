// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_LSTMUTIL_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_LSTMUTIL_HPP_

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

#endif // POPART_WILLOW_INCLUDE_POPART_OP_LSTMUTIL_HPP_
