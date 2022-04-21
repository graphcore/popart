// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART__INTERNAL_IR_BINDINGS_TRANSFORM_AUTODIFF_HPP
#define POPART__INTERNAL_IR_BINDINGS_TRANSFORM_AUTODIFF_HPP

#include <pybind11/pybind11.h>
#include <popart/bwdgraphinfo.hpp>
#include <popart/transforms/autodiff.hpp>
#include <popart/transforms/transform.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {
namespace transforms {

/**
 * Add bindings for `popart::Autodiff` class to pybind module.
 **/
void bindAutodiff(py::module &m);

} // namespace transforms
} // namespace ir
} // namespace _internal
} // namespace popart

#endif