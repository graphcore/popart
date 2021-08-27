// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/basicoptionals.hpp"
#include "bindings/debugcontext.hpp"
#include "bindings/graph.hpp"
#include "bindings/graphid.hpp"
#include "bindings/ir.hpp"
#include "bindings/op.hpp"
#include "bindings/op/_all.hpp"
#include "bindings/op/enums.hpp"
#include "bindings/opdebuginfo.hpp"
#include "bindings/opidentifier.hpp"
#include "bindings/scope.hpp"
#include "bindings/tensor.hpp"
#include "bindings/tensordata.hpp"
#include "bindings/tensorinfo.hpp"
#include "bindings/tensorlocation.hpp"
#include "bindings/tensors.hpp"

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

  bindDebugContext(m);
  bindGraph(m);
  bindGraphId(m);
  bindIr(m);
  bindOp(m);
  bindScope(m);
  bindTensor(m);
  bindTensorData(m);
  bindTensorInfo(m);
  bindTensors(m);
  bindBasicOptionals(m);
  bindOpDebugInfo(m);
  bindOpIdentifier(m);
  bindTensorLocation(m);
  {
    op::_bindAll(m);
    op::bindEnums(m);
  }
}
} // namespace ir
} // namespace _internal
} // namespace popart
