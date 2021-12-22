// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <popart/op/lstmutil.hpp>

namespace popart {

ActivationFunction fromString(const std::string &s) {
  if (s == "Sigmoid") {
    return ActivationFunction::Sigmoid;
  } else if (s == "Relu") {
    return ActivationFunction::Relu;
  } else if (s == "Tanh") {
    return ActivationFunction::Tanh;
  } else if (s == "Gelu") {
    return ActivationFunction::Gelu;
  } else if (s == "Swish") {
    return ActivationFunction::Swish;
  } else if (s == "Softmax") {
    return ActivationFunction::Softmax;
  } else if (s == "SoftmaxStable") {
    return ActivationFunction::SoftmaxStable;
  } else if (s == "SoftmaxScaled") {
    return ActivationFunction::SoftmaxScaled;
  } else {
    return ActivationFunction::Invalid;
  }
}

std::ostream &operator<<(std::ostream &os, const ActivationFunction &af) {
  switch (af) {
  case ActivationFunction::Sigmoid:
    os << "Sigmoid";
    break;
  case ActivationFunction::Relu:
    os << "Relu";
    break;
  case ActivationFunction::Tanh:
    os << "Tanh";
    break;
  case ActivationFunction::Gelu:
    os << "Gelu";
    break;
  case ActivationFunction::Swish:
    os << "Swish";
    break;
  case ActivationFunction::Softmax:
    os << "Softmax";
    break;
  case ActivationFunction::SoftmaxStable:
    os << "SoftmaxStable";
    break;
  case ActivationFunction::SoftmaxScaled:
    os << "SoftmaxScaled";
    break;
  case ActivationFunction::Invalid:
  default:
    os << "Invalid";
    break;
  }
  return os;
}

} // namespace popart