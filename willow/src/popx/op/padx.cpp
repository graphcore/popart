// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <string>
#include <utility>
#include <vector>
#include <poplar/Tensor.hpp>
#include <popops/Pad.hpp>
#include <popops/Zero.hpp>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/op/pad.hpp>
#include <popart/op/slice.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/padgradx.hpp>
#include <popart/popx/op/padx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensornames.hpp>

#include "popart/logging.hpp"
#include "popart/op.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/popx/linearmapper.hpp"
#include "popart/popx/op/slicex.hpp"
#include "popart/popx/popopx.hpp"
#include "popart/popx/poptensors.hpp"
#include "popart/tensor.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensors.hpp"

namespace popart {
namespace popx {

const BasePadOp &BasePadOpx::getBasePadOp() const {
  return dynamic_cast<const BasePadOp &>(*op_p);
}

PadOpx::PadOpx(Op *op, Devicex *devicex) : BasePadOpx(op, devicex) {
  verifyOp<BasePadOutplaceOp>(op);
}

BasePadOpx::BasePadOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<BasePadOp>(op);
}

std::pair<bool, snap::Tensor> BasePadOpx::getPropitiousPadLayout() const {

  auto &&bop = getBasePadOp();

  // The SliceGrad and its corresponding non-grad op:
  //
  // act_in       act_in_grad
  //    |            |
  //  Slice       SliceGrad
  //    |            |
  // act_out      act_out_grad
  //
  // We want to find act_in.
  //
  // 1. From 'act_out_grad', find 'act_out'
  // 2. From 'act_out', find its producer
  // 3. From its producer, find 'act_in'
  snap::Tensor dummy;

  auto inId = bop.inId(BasePadOp::getInIndex());
  if (!isGradId(inId)) {
    return {false, dummy};
  }

  TensorId act_out = getNonGradId(inId);

  if (!dv_p->lowering().tensors().contains(act_out)) {
    return {false, dummy};
  }

  Tensor *act_out_t = bop.getGraph().getTensors().get(act_out);
  if (!act_out_t->hasProducer()) {
    return {false, dummy};
  }

  Op *sliceOp = act_out_t->getProducer();
  // If a transform has modified the Ir by here, give up.
  if (!dynamic_cast<BaseSliceOp *>(sliceOp)) {
    return {false, dummy};
  }

  TensorId act_in = sliceOp->inId(BaseSliceOp::getInIndex());
  if (!dv_p->lowering().tensors().contains(act_in)) {
    return {false, dummy};
  }

  // Check the shape.
  // In the case where the PadSum transform converts padsum into concat, this
  // might not match.
  const auto expectedShape = bop.outShape(BasePadOp::getOutIndex());
  std::vector<size_t> expectedShape_u64;
  expectedShape_u64.reserve(expectedShape.size());
  for (auto d : expectedShape) {
    expectedShape_u64.push_back(static_cast<size_t>(d));
  }
  const auto propitiousLayoutTensor = get(act_in).getPoplarTensor();
  if (expectedShape_u64 != propitiousLayoutTensor.shape()) {
    return {false, dummy};
  }

  logging::devicex::debug(
      "Found a propitious tile mapping for the Pad's output, "
      "based on already tile-mapped lowering().Tensors, for op {}",
      bop.str());

  // Found a good layout!

  // return {false, dummy};
  return {true, get(act_in)};
}

BasePadOpx::Chisseled BasePadOpx::getChisseled(const snap::Tensor &t0) const {

  auto t             = t0.getPoplarTensor();
  const size_t tRank = t.rank();
  auto &&op          = getBasePadOp();

  const auto lowerPadding = op.getLowerPadding();
  const auto upperPadding = op.getUpperPadding();

  std::vector<snap::Tensor> lowPads(tRank);
  std::vector<snap::Tensor> uppPads(tRank);

  for (size_t d = 0; d < tRank; ++d) {
    const auto d_u32   = static_cast<uint32_t>(d);
    const auto dimSize = t.dim(d_u32);
    const auto low     = lowerPadding[d];
    const auto upp     = dimSize - upperPadding[d];
    lowPads[d]         = snap::Tensor{t.slice(0, low, d_u32), graph()};
    uppPads[d]         = snap::Tensor{t.slice(upp, dimSize, d_u32), graph()};
    t                  = t.slice(low, upp, d_u32);
  }
  return {snap::Tensor{t, graph()}, lowPads, uppPads};
}

snap::Tensor BasePadOpx::cloneNcopyEdges(snap::Tensor t,
                                         snap::program::Sequence &se) const {

  logging::devicex::debug("Cloning and copying the constant padding for op {}",
                          getBasePadOp().str());

  const auto tRank = t.rank();

  // partition the input into the "core" and the padding edges.
  auto chisseled = getChisseled(t);
  t              = chisseled.core;
  auto leftPads  = chisseled.lows;
  auto rightPads = chisseled.upps;

  // clone and copy the edges
  // These copies will be merged by poplar, so it is not necessary to
  // concatenate all the lowering().Tensors and create one single large copy
  // program.
  for (auto &p : leftPads) {
    if (p.numElements() > 0) {
      p = cloneNcopy(se, p);
    }
  }
  for (auto &p : rightPads) {
    if (p.numElements() > 0) {
      p = cloneNcopy(se, p);
    }
  }

  // wrap the core in the copied edges. Thus the core remains an alias od the
  // input.
  auto pt = t.getPoplarTensor();
  for (int d = static_cast<int>(tRank) - 1; d >= 0; --d) {
    const auto d_u64 = static_cast<size_t>(d);
    const auto d_u32 = static_cast<uint32_t>(d);

    if (leftPads[d_u64].numElements() > 0) {
      pt = poplar::concat({leftPads[d_u64].getPoplarTensor(), pt}, d_u32);
    }
    if (rightPads[d_u64].numElements() > 0) {
      pt = poplar::concat({pt, rightPads[d_u64].getPoplarTensor()}, d_u32);
    }
  }

  return snap::Tensor{pt, graph()};
}

snap::Tensor BasePadOpx::constantModePadGrow(snap::Tensor inTensor,
                                             snap::program::Sequence &s,
                                             bool inPlaceAllowed) const {

  auto mk_padded =
      [this, &s, &inTensor, inPlaceAllowed](
          const popops::padding::MappingMethod mappingMethod) -> snap::Tensor {
    auto &&padBaseOp        = getBasePadOp();
    const auto lowerPadding = padBaseOp.getLowerPadding();
    const auto upperPadding = padBaseOp.getUpperPadding();
    const auto padValue     = padBaseOp.getPadValue();

    if (!inPlaceAllowed) {
      inTensor = cloneNcopy(s, inTensor);
    }

    return snap::Tensor{popops::pad(PopOpx::graph().getPoplarGraph(),
                                    inTensor.getPoplarTensor(),
                                    lowerPadding,
                                    upperPadding,
                                    padValue,
                                    mappingMethod),
                        PopOpx::graph()};
  };

  // If padding a tensor with a dim of size 0, the original tensor must have no
  // elements, so "edge" mapping is not possible and we must do it ourselves.
  // Note a tensor has 0 elements iff it has a zero-sized dimension.
  if (inTensor.numElements() == 0) {
    auto outTensor = mk_padded(popops::padding::MappingMethod::NONE);
    dv_p->lowering().getLinearMapper().mapTensor(PopOpx::graph(), outTensor);
    return outTensor;
  }

  // Can we find a good layout for the constant padding?
  // This approach is never inplace, as the full input is copied. So we do not
  // need to cloneNcopy it.
  const auto propitious = getPropitiousPadLayout();
  if (propitious.first) {
    auto outTensor    = graph().clone(propitious.second);
    auto chisseledDst = getChisseled(outTensor);
    s.getPoplarSequence().add(snap::program::Copy(
        inTensor, chisseledDst.core, false, debugContext()));
    std::vector<poplar::Tensor> allPads;
    allPads.reserve(chisseledDst.lows.size() + chisseledDst.upps.size());
    for (const auto &x : chisseledDst.lows) {
      allPads.push_back(x.flatten().getPoplarTensor());
    }
    for (const auto &x : chisseledDst.upps) {
      allPads.push_back(x.flatten().getPoplarTensor());
    }
    auto cat = poplar::concat(allPads, 0);
    popops::zero(graph().getPoplarGraph(),
                 cat,
                 s.getPoplarSequence(),
                 debugContext("zero"));
    return outTensor;
  }

  else {
    auto outTensor = mk_padded(popops::padding::MappingMethod::EDGE);
    if (outTensor.getPoplarTensor().containsAliases()) {
      outTensor = cloneNcopyEdges(outTensor, s);
    }
    return outTensor;
  }
}

snap::Tensor BasePadOpx::unflippedPadGrow(snap::Tensor inTensor,
                                          snap::program::Sequence &prog,
                                          bool inPlaceAllowed) const {

  auto &&padBaseOp = getBasePadOp();
  const auto mode  = padBaseOp.getMode();

  if (mode == "constant") {
    return constantModePadGrow(inTensor, prog, inPlaceAllowed);
  }

  if (!inPlaceAllowed) {
    inTensor = cloneNcopy(prog, inTensor);
  }

  const auto lp = padBaseOp.getLowerPadding();
  const auto up = padBaseOp.getUpperPadding();

  if (mode == "edge") {
    return snap::Tensor{
        popops::pad(
            inTensor.getPoplarTensor(), lp, up, popops::padding::Type::EDGE),
        graph()};
  }

  if (mode == "reflect") {
    return snap::Tensor{
        popops::pad(
            inTensor.getPoplarTensor(), lp, up, popops::padding::Type::REFLECT),
        graph()};
  }

  std::ostringstream oss;
  oss << "Invalid pad mode `" << mode
      << "' in padGrow. Expected one of (edge, reflect, constant)";
  throw error(oss.str());
}

snap::Tensor BasePadOpx::padGrow(snap::Tensor inTensor,
                                 snap::program::Sequence &prog,
                                 bool inPlaceAllowed) const {
  return unflippedPadGrow(flip(inTensor), prog, inPlaceAllowed);
}

void PadOpx::grow(snap::program::Sequence &prog) const {
  auto in0    = PopOpx::getInTensor(BasePadOp::getInIndex());
  auto padded = padGrow(in0, prog, false);
  setOutTensor(BasePadOp::getOutIndex(), padded);
}

void PadInplaceOpx::grow(snap::program::Sequence &prog) const {
  auto in0    = PopOpx::getInTensor(BasePadOp::getInIndex());
  auto padded = padGrow(in0, prog, true);
  setOutTensor(BasePadOp::getOutIndex(), padded);
}

PadGradOpx::PadGradOpx(Op *op, Devicex *devicex) : SliceOpx(op, devicex) {
  verifyOp<PadGradOpx>(op, Onnx::GradOperators::PadGrad);
}

snap::Tensor BasePadOpx::flip(const snap::Tensor &inTensor) const {
  auto oTensor     = inTensor.getPoplarTensor();
  auto &&padBaseOp = getBasePadOp();
  for (auto f : padBaseOp.getFlips()) {
    oTensor = oTensor.reverse(f);
  }
  return snap::Tensor{oTensor, graph()};
}

PadInplaceOpx::PadInplaceOpx(Op *op, Devicex *devicex)
    : BasePadOpx(op, devicex) {
  verifyOp<PadInplaceOp>(op, Onnx::CustomOperators::PadInplace);
}

namespace {
OpxCreator<PadOpx> padOpxCreator({Onnx::Operators::Pad_2,
                                  Onnx::Operators::Pad_11});

OpxCreator<PadGradOpx> padGradOpxCreator(Onnx::GradOperators::PadGrad);

OpxCreator<PadInplaceOpx>
    padxInplaceOpxCreator(Onnx::CustomOperators::PadInplace);

OpxCreator<PadOpx> sliceGradOpxCreator(Onnx::GradOperators::SliceGrad);
} // namespace

} // namespace popx
} // namespace popart
