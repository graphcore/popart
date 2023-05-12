// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popnn/NonLinearityDef.hpp>
#include <popart/error.hpp>
#include <popart/popx/op/lstmxutil.hpp>

#include "popart/logging.hpp"
#include "popart/op/lstmutil.hpp"

namespace popart {
namespace popx {

popnn::NonLinearityType convert(ActivationFunction af) {
  switch (af) {
  case ActivationFunction::Sigmoid:
    return popnn::NonLinearityType::SIGMOID;
  case ActivationFunction::Relu:
    return popnn::NonLinearityType::RELU;
  case ActivationFunction::Tanh:
    return popnn::NonLinearityType::TANH;
  case ActivationFunction::Gelu:
    return popnn::NonLinearityType::GELU;
  case ActivationFunction::GeluErf:
    return popnn::NonLinearityType::GELU_ERF;
  case ActivationFunction::Swish:
    return popnn::NonLinearityType::SWISH;
  case ActivationFunction::Softmax:
    return popnn::NonLinearityType::SOFTMAX;
  case ActivationFunction::SoftmaxStable:
    return popnn::NonLinearityType::SOFTMAX_STABLE;
  case ActivationFunction::SoftmaxScaled:
    return popnn::NonLinearityType::SOFTMAX_SCALED;
  default:
    throw error("Poplibs lstm does not support activation function '{}'", af);
  }
}

} // namespace popx
} // namespace popart
