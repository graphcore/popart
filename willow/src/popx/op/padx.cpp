// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popops/Pad.hpp>
#include <popops/Zero.hpp>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/names.hpp>
#include <popart/op/pad.hpp>
#include <popart/op/slice.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/padgradx.hpp>
#include <popart/popx/op/padx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensornames.hpp>
#include <popart/util.hpp>

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

std::pair<bool, poplar::Tensor> BasePadOpx::getPropitiousPadLayout() const {

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
  poplar::Tensor dummy;

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
  const auto propitiousLayoutTensor = get(act_in);
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

BasePadOpx::Chisseled BasePadOpx::getChisseled(const poplar::Tensor &t0) const {

  auto t             = t0;
  const size_t tRank = t.rank();
  auto &&op          = getBasePadOp();

  const auto lowerPadding = op.getLowerPadding();
  const auto upperPadding = op.getUpperPadding();

  std::vector<poplar::Tensor> lowPads(tRank);
  std::vector<poplar::Tensor> uppPads(tRank);

  for (size_t d = 0; d < tRank; ++d) {
    const auto d_u32   = static_cast<uint32_t>(d);
    const auto dimSize = t.dim(d_u32);
    const auto low     = lowerPadding[d];
    const auto upp     = dimSize - upperPadding[d];
    lowPads[d]         = t.slice(0, low, d_u32);
    uppPads[d]         = t.slice(upp, dimSize, d_u32);
    t                  = t.slice(low, upp, d_u32);
  }
  return {t, lowPads, uppPads};
}

poplar::Tensor
BasePadOpx::cloneNcopyEdges(poplar::Tensor t,
                            poplar::program::Sequence &se) const {

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
  for (int d = static_cast<int>(tRank) - 1; d >= 0; --d) {
    const auto d_u64 = static_cast<size_t>(d);
    const auto d_u32 = static_cast<uint32_t>(d);

    if (leftPads[d_u64].numElements() > 0) {
      t = poplar::concat({leftPads[d_u64], t}, d_u32);
    }
    if (rightPads[d_u64].numElements() > 0) {
      t = poplar::concat({t, rightPads[d_u64]}, d_u32);
    }
  }

  return t;
}

poplar::Tensor BasePadOpx::constantModePadGrow(poplar::Tensor inTensor,
                                               poplar::program::Sequence &s,
                                               bool inPlaceAllowed) const {

  auto mk_padded = [this, &s, &inTensor, inPlaceAllowed](
                       const popops::padding::MappingMethod mappingMethod)
      -> poplar::Tensor {
    auto &&padBaseOp        = getBasePadOp();
    const auto lowerPadding = padBaseOp.getLowerPadding();
    const auto upperPadding = padBaseOp.getUpperPadding();
    const auto padValue     = padBaseOp.getPadValue();

    if (!inPlaceAllowed) {
      inTensor = cloneNcopy(s, inTensor);
    }

    return popops::pad(PopOpx::graph().getPoplarGraph(),
                       inTensor,
                       lowerPadding,
                       upperPadding,
                       padValue,
                       mappingMethod);
  };

  // If padding a tensor with a dim of size 0, the original tensor must have no
  // elements, so "edge" mapping is not possible and we must do it ourselves.
  // Note a tensor has 0 elements iff it has a zero-sized dimension.
  if (inTensor.numElements() == 0) {
    auto outTensor = snap::Tensor{
        mk_padded(popops::padding::MappingMethod::NONE), PopOpx::graph()};
    dv_p->lowering().getLinearMapper().mapTensor(PopOpx::graph(), outTensor);
    return outTensor.getPoplarTensor();
  }

  // Can we find a good layout for the constant padding?
  // This approach is never inplace, as the full input is copied. So we do not
  // need to cloneNcopy it.
  const auto propitious = getPropitiousPadLayout();
  if (propitious.first) {
    auto outTensor    = graph().getPoplarGraph().clone(propitious.second);
    auto chisseledDst = getChisseled(outTensor);
    s.add(poplar::program::Copy(
        inTensor, chisseledDst.core, false, debugContext()));
    auto allPads = chisseledDst.lows;
    allPads.insert(
        allPads.end(), chisseledDst.upps.cbegin(), chisseledDst.upps.cend());
    for (auto &x : allPads) {
      x = x.flatten();
    }
    auto cat = poplar::concat(allPads, 0);
    popops::zero(graph().getPoplarGraph(), cat, s, debugContext("zero"));
    return outTensor;
  }

  else {
    auto outTensor = mk_padded(popops::padding::MappingMethod::EDGE);
    if (outTensor.containsAliases()) {
      outTensor = cloneNcopyEdges(outTensor, s);
    }
    return outTensor;
  }
}

poplar::Tensor BasePadOpx::unflippedPadGrow(poplar::Tensor inTensor,
                                            poplar::program::Sequence &prog,
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
    return popops::pad(inTensor, lp, up, popops::padding::Type::EDGE);
  }

  if (mode == "reflect") {
    return popops::pad(inTensor, lp, up, popops::padding::Type::REFLECT);
  }

  std::ostringstream oss;
  oss << "Invalid pad mode `" << mode
      << "' in padGrow. Expected one of (edge, reflect, constant)";
  throw error(oss.str());
}

poplar::Tensor BasePadOpx::padGrow(poplar::Tensor inTensor,
                                   poplar::program::Sequence &prog,
                                   bool inPlaceAllowed) const {
  return unflippedPadGrow(flip(inTensor), prog, inPlaceAllowed);
}

void PadOpx::grow(poplar::program::Sequence &prog) const {
  auto in0    = PopOpx::getInTensor(BasePadOp::getInIndex());
  auto padded = padGrow(in0, prog, false);
  setOutTensor(BasePadOp::getOutIndex(), padded);
}

void PadInplaceOpx::grow(poplar::program::Sequence &prog) const {
  auto in0    = PopOpx::getInTensor(BasePadOp::getInIndex());
  auto padded = padGrow(in0, prog, true);
  setOutTensor(BasePadOp::getOutIndex(), padded);
}

PadGradOpx::PadGradOpx(Op *op, Devicex *devicex) : SliceOpx(op, devicex) {
  verifyOp<PadGradOpx>(op, Onnx::GradOperators::PadGrad);
}

poplar::Tensor BasePadOpx::flip(const poplar::Tensor &inTensor) const {
  auto oTensor     = inTensor;
  auto &&padBaseOp = getBasePadOp();
  for (auto f : padBaseOp.getFlips()) {
    oTensor = oTensor.reverse(f);
  }
  return oTensor;
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
