// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART_PYTHON_POPART__INTERNAL_IR_BINDINGS_OP_CONCAT_HPP_
#define POPART_PYTHON_POPART__INTERNAL_IR_BINDINGS_OP_CONCAT_HPP_

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {
namespace op {

/**
 * Add bindings for the concat op.
 **/
void bindConcat(py::module &m);

} // namespace op
} // namespace ir
} // namespace _internal
} // namespace popart

#endif // POPART_PYTHON_POPART__INTERNAL_IR_BINDINGS_OP_CONCAT_HPP_
