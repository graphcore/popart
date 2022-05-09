// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <string>

#include "bindings/basicoptionals.hpp"
#include "bindings/bwdgraphinfo.hpp"
#include "bindings/commgroup.hpp"
#include "bindings/debugcontext.hpp"
#include "bindings/graph.hpp"
#include "bindings/graphid.hpp"
#include "bindings/ir.hpp"
#include "bindings/op.hpp"
#include "bindings/op/_all.hpp"
#include "bindings/op/accumulate.hpp"
#include "bindings/op/accumulatorscale.hpp"
#include "bindings/op/accumulatorzero.hpp"
#include "bindings/op/adamupdater.hpp"
#include "bindings/op/argminmax.hpp"
#include "bindings/op/call.hpp"
#include "bindings/op/concat.hpp"
#include "bindings/op/conv.hpp"
#include "bindings/op/enums.hpp"
#include "bindings/op/ipucopy.hpp"
#include "bindings/op/loop.hpp"
#include "bindings/op/matmul.hpp"
#include "bindings/op/optimizervalue.hpp"
#include "bindings/op/optional.hpp"
#include "bindings/op/pool.hpp"
#include "bindings/op/roialign.hpp"
#include "bindings/op/varupdate.hpp"
#include "bindings/opidentifier.hpp"
#include "bindings/region.hpp"
#include "bindings/remotebufferinfo.hpp"
#include "bindings/scope.hpp"
#include "bindings/tensor.hpp"
#include "bindings/tensordata.hpp"
#include "bindings/tensorinfo.hpp"
#include "bindings/tensorlocation.hpp"
#include "bindings/tensors.hpp"
#include "bindings/topocons.hpp"
#include "bindings/util.hpp"
// transforms
#include "bindings/transforms/autodiff.hpp"
#include "bindings/transforms/mergeexchange.hpp"
#include "bindings/transforms/prune.hpp"
#include "bindings/transforms/transform.hpp"
// Patterns
#include "bindings/patterns/pattern.hpp"
#include "bindings/patterns/patterns.hpp"

namespace popart {
namespace _internal {
namespace ir {

PYBIND11_MODULE(popart_internal_ir, m) {
  m.doc() = "This module is an internal PopART API (`popart._internal.ir`) "
            "that is used to implement the public PopXL API. This "
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
  bindRegion(m);
  bindTensorLocation(m);
  bindRemoteBufferInfo(m);
  bindBwdGraphInfo(m);
  bindCommGroup(m);
  bindUtil(m);
  bindTopoCons(m);
  // Ops
  {
    op::_bindAll(m);
    op::bindCall(m);
    op::bindEnums(m);
    op::bindOptional(m);
    op::bindIpuCopy(m);
    op::bindMatmul(m);
    op::bindRepeat(m);
    op::bindOptimizerValue(m);
    op::bindVarupdate(m);
    op::bindAccumulate(m);
    op::bindAccumulatorScale(m);
    op::bindAccumulatorZero(m);
    op::bindAdamUpdater(m);
    op::bindConcat(m);
    op::bindConv(m);
    op::bindRoiAlign(m);
    op::bindPool(m);
    op::bindArgMinMax(m);
  }
  // Transforms
  {
    auto sm = m.def_submodule("transforms");
    transforms::bindTransform(sm);
    transforms::bindPrune(sm);
    transforms::bindAutodiff(sm);
    transforms::bindMergeExchange(sm);
  }
  // Patterns
  {
    auto sm = m.def_submodule("patterns");
    patterns::bindPattern(sm);
    patterns::bindPatterns(sm);
  }
}
} // namespace ir
} // namespace _internal
} // namespace popart
