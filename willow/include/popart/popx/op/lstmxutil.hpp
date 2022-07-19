// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_LSTMXUTIL_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_LSTMXUTIL_HPP_

#include <popnn/NonLinearityDef.hpp>
#include <popart/op/lstmutil.hpp>

namespace popart {
namespace popx {

popnn::NonLinearityType convert(ActivationFunction);

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_LSTMXUTIL_HPP_
