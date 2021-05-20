// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LSTMXUTIL_HPP
#define GUARD_NEURALNET_LSTMXUTIL_HPP
#include <popart/op/lstm.hpp>

#include <popnn/NonLinearity.hpp>

namespace popart {
namespace popx {

popnn::NonLinearityType convert(ActivationFunction af);

} // namespace popx
} // namespace popart

#endif
