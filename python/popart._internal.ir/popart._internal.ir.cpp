// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/graph.hpp"
#include "bindings/graphid.hpp"
#include "bindings/ir.hpp"

#include <pybind11/pybind11.h>

#include <popart/ir.hpp>

namespace popart {
namespace _internal {
namespace ir {

PYBIND11_MODULE(popart_internal_ir, m) {
  m.doc() = "This module is an internal PopART API (`popart._internal.ir`) "
            "that is used to implement the public `popart.ir` API. This "
            "internal API is not intended for public use and may change "
            "between releases with no guarantee of backwards compatibility "
            "or deprecation periods.";

  bindGraph(m);
  bindGraphId(m);
  bindIr(m);
}

} // namespace ir
} // namespace _internal
} // namespace popart