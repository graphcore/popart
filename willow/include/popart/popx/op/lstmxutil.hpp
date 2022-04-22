// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LSTMXUTIL_HPP
#define GUARD_NEURALNET_LSTMXUTIL_HPP

#include <popnn/NonLinearityDef.hpp>
#include <popart/op/lstmutil.hpp>

namespace popart {
namespace popx {

popnn::NonLinearityType convert(ActivationFunction);

} // namespace popx
} // namespace popart

#endif
