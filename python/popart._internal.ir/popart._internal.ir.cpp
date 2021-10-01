// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/basicoptionals.hpp"
#include "bindings/bwdgraphinfo.hpp"
#include "bindings/debugcontext.hpp"
#include "bindings/graph.hpp"
#include "bindings/graphid.hpp"
#include "bindings/ir.hpp"
#include "bindings/op.hpp"
#include "bindings/op/_all.hpp"
#include "bindings/op/call.hpp"
#include "bindings/op/enums.hpp"
#include "bindings/op/ipucopy.hpp"
#include "bindings/op/manualbindops.hpp"
#include "bindings/op/matmul.hpp"
#include "bindings/op/optional.hpp"
#include "bindings/opidentifier.hpp"
#include "bindings/scope.hpp"
#include "bindings/tensor.hpp"
#include "bindings/tensordata.hpp"
#include "bindings/tensorinfo.hpp"
#include "bindings/tensorlocation.hpp"
#include "bindings/tensors.hpp"
#include "bindings/util.hpp"
// transforms
#include "bindings/transforms/autodiff.hpp"
#include "bindings/transforms/prune.hpp"
#include "bindings/transforms/transform.hpp"

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
  bindOpIdentifier(m);
  bindTensorLocation(m);
  bindBwdGraphInfo(m);
  bindUtil(m);
  // Ops
  {
    op::_bindAll(m);
    op::bindCall(m);
    op::bindEnums(m);
    op::bindOptional(m);
    op::bindIpuCopy(m);
    op::bindMatmul(m);
  }
  // Transforms
  {
    auto sm = m.def_submodule("transforms");
    transforms::bindTransform(sm);
    transforms::bindPrune(sm);
    transforms::bindAutodiff(sm);
  }
}
} // namespace ir
} // namespace _internal
} // namespace popart
