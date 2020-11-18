// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_POPEXECUTABLESERIALIZATION_HPP
#define GUARD_NEURALNET_POPEXECUTABLESERIALIZATION_HPP

#include <map>
#include <memory>
#include <set>

#include <iostream>
#include <popart/popx/irlowering.hpp>

namespace popart {
namespace popx {
namespace serialization {

void serializeExecutable(std::ostream &out,
                         const popart::popx::Executablex &executable);

std::unique_ptr<popart::popx::Executablex>
deserializeExecutable(std::istream &in,
                      popart::Ir &ir,
                      popart::popx::IrLowering &lowering);

} // namespace serialization
} // namespace popx
} // namespace popart

#endif // GUARD_NEURALNET_WILLOWEXECUTABLESERIALIZATION_HPP
