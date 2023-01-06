// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <vector>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <popart/op/tensorremap.hpp>
#include <popart/popx/op/tensorremapx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/names.hpp"
#include "popart/popx/opx.hpp"
#include "popart/region.hpp" // IWYU pragma: keep

namespace popart {
class Op;

namespace popx {
class Devicex;

TensorRemapOpx::TensorRemapOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<TensorRemapOp>(op, Onnx::CustomOperators::TensorRemap_1);
}

void TensorRemapOpx::grow(poplar::program::Sequence &prog) const {
  poplar::Tensor out;

  if (hasInput(TensorRemapOp::getRefInIndex())) {
    // Clone from reference
    out = graph().clone(getInTensor(hasInput(TensorRemapOp::getRefInIndex())),
                        debugContext(outId(TensorRemapOp::getOutIndex())));
    setOutTensor(TensorRemapOp::getOutIndex(), out);
  } else {
    // Create new output tensor (externally)
    out = getOutTensor(TensorRemapOp::getOutIndex());
  }

  poplar::program::Copy copyProg(
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

poplar::Tensor TensorRemapOpx::unwindTensorLayout(poplar::Tensor tensor,
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
