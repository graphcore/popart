// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <vector>
#include <poplar/Graph.hpp>
#include <popart/op/tensorremap.hpp>
#include <popart/popx/op/tensorremapx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/names.hpp"
#include "popart/popx/popopx.hpp"
#include "popart/region.hpp" // IWYU pragma: keep

namespace popart {
class Op;

namespace popx {
class Devicex;

TensorRemapOpx::TensorRemapOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<TensorRemapOp>(op, Onnx::CustomOperators::TensorRemap_1);
}

void TensorRemapOpx::grow(snap::program::Sequence &prog) const {
  snap::Tensor out;

  if (hasInput(TensorRemapOp::getRefInIndex())) {
    // Clone from reference
    out = snap::Tensor{graph().getPoplarGraph().clone(
                           getInTensor(hasInput(TensorRemapOp::getRefInIndex()))
                               .getPoplarTensor(),
                           debugContext(outId(TensorRemapOp::getOutIndex()))),
                       graph()};
    setOutTensor(TensorRemapOp::getOutIndex(), out);
  } else {
    // Create new output tensor (externally)
    out = getOutTensor(TensorRemapOp::getOutIndex());
  }

  snap::program::Copy copyProg(
      getInTensor(TensorRemapOp::getInIndex()),
      out,
      false,
      debugContext(outId(TensorRemapOp::getOutIndex())));
  prog.add(copyProg);
}

bool TensorRemapOpx::outputCreatedExternally(OutIndex) const {
  if (hasInput(TensorRemapOp::getRefInIndex())) {
    // False, tensor will be cloned from reference
    return false;
  } else {
    // True, tensor creator searched externally
    return true;
  }
}

InputCreatorType TensorRemapOpx::getInputCreatorType(InIndex index) const {
  if (index == TensorRemapOp::getInIndex()) {
    return InputCreatorType::CanUnwind;
  } else {
    return InputCreatorType::Deadend;
  }
}

snap::Tensor TensorRemapOpx::unwindTensorLayout(snap::Tensor tensor,
                                                InIndex,
                                                OutIndex) const {
  return tensor;
}

view::RegMap TensorRemapOpx::unwindRegion(InIndex, OutIndex) const {
  return [](const view::Region &r) { return view::Regions(1, r); };
}

namespace {
OpxCreator<TensorRemapOpx>
    TensorRemapOpxCreator(Onnx::CustomOperators::TensorRemap_1);
} // namespace
} // namespace popx
} // namespace popart
