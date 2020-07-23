// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popops/Pad.hpp>
#include <popops/Zero.hpp>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/names.hpp>
#include <popart/op/pad.hpp>
#include <popart/op/slice.hpp>
#include <popart/popx/devicex.hpp>
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

BasePadOpx::BasePadOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
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

  if (!dv_p->tensors.contains(act_out)) {
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
  if (!dv_p->tensors.contains(act_in)) {
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
      "based on already tile-mapped Tensors, for op {}",
      bop.str());

  // Found a good layout!

  // return {false, dummy};
  return {true, get(act_in)};
}

poplar::Tensor
BasePadOpx::padWithTargetMapping(const poplar::Tensor &toPad,
                                 const poplar::Tensor &toMapEdgesFrom) const {

  auto &&op = getBasePadOp();

  const auto lowerPadding = op.getLowerPadding();
  const auto upperPadding = op.getUpperPadding();
  const auto padValue     = op.getPadValue();

  const size_t tRank = lowerPadding.size();
  if (toPad.rank() != tRank || toMapEdgesFrom.rank() != tRank) {
    throw internal_error("Failed size comparison in getEdgeConstClone (1)");
  }

  for (uint32_t i = 0; i < tRank; ++i) {
    if (toPad.dim(i) + lowerPadding[i] + upperPadding[i] !=
        toMapEdgesFrom.dim(i)) {
      throw internal_error("Failed size comparison in getEdgeConstClone (2)");
    }
  }

  // return a Tensor which has the same tile mapping as x, but is constant of
  auto getConstPad = [this, padValue](const poplar::Tensor &x,
                                      const std::string &dbs) {
    auto constPad = graph().addConstant<float>(
        x.elementType(),
        x.shape(),
        std::vector<float>(x.numElements(), padValue),
        dbs);
    graph().setTileMapping(constPad, graph().getTileMapping(x));
    return constPad;
  };

  std::vector<poplar::Tensor> leftPads(tRank);
  std::vector<poplar::Tensor> rightPads(tRank);
  auto chisseledToEmulate = toMapEdgesFrom;

  for (size_t d = 0; d < tRank; ++d) {
    const auto d_u32   = static_cast<uint32_t>(d);
    const auto dimSize = chisseledToEmulate.dim(d_u32);
    const auto low     = lowerPadding[d];
    const auto upp     = dimSize - upperPadding[d];

    const auto lowToClone = chisseledToEmulate.slice(0, low, d_u32);
    const auto uppToClone = chisseledToEmulate.slice(upp, dimSize, d_u32);
    leftPads[d]  = getConstPad(lowToClone, "/leftPad_dim" + std::to_string(d));
    rightPads[d] = getConstPad(uppToClone, "/rightPad_dim" + std::to_string(d));
    chisseledToEmulate = chisseledToEmulate.slice(low, upp, d_u32);
  }

  auto padded = toPad;

  for (int d = static_cast<int>(tRank) - 1; d >= 0; --d) {
    const auto d_u64 = static_cast<size_t>(d);
    const auto d_u32 = static_cast<uint32_t>(d);

    if (leftPads[d_u64].numElements() > 0) {
      padded = poplar::concat({leftPads[d_u64], padded}, d_u32);
    }
    if (rightPads[d_u64].numElements() > 0) {
      padded = poplar::concat({padded, rightPads[d_u64]}, d_u32);
    }
  }

  if (padded.shape() != toMapEdgesFrom.shape()) {
    throw internal_error("padded should be of the same shape as the original");
  }
  return padded;
}

poplar::Tensor
BasePadOpx::constantModePadGrow(poplar::Tensor inTensor,
                                poplar::program::Sequence &) const {

  auto mk_padded =
      [this, &inTensor](const popops::padding::MappingMethod mappingMethod)
      -> poplar::Tensor {
    auto &&padBaseOp        = getBasePadOp();
    const auto lowerPadding = padBaseOp.getLowerPadding();
    const auto upperPadding = padBaseOp.getUpperPadding();
    const auto padValue     = padBaseOp.getPadValue();

    return popops::pad(Opx::graph(),
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
    auto outTensor = mk_padded(popops::padding::MappingMethod::NONE);

    dv_p->getLinearMapper().mapTensor(Opx::graph(), outTensor);

    return outTensor;
  }

  // Can we find a good layout for the constant padding?
  auto propitious = getPropitiousPadLayout();
  if (propitious.first) {
    return padWithTargetMapping(inTensor, propitious.second);
  }

  else {
    return mk_padded(popops::padding::MappingMethod::EDGE);
  }
}

poplar::Tensor BasePadOpx::padGrow(poplar::Tensor inTensor,
                                   poplar::program::Sequence &prog) const {

  auto &&padBaseOp = getBasePadOp();
  const auto mode  = padBaseOp.getMode();

  if (mode == "constant") {
    return constantModePadGrow(inTensor, prog);
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

void PadOpx::grow(poplar::program::Sequence &prog) const {
  auto in0      = Opx::getInTensor(BasePadOp::getInIndex());
  auto inTensor = cloneNcopy(prog, in0);

  auto &&padBaseOp = getBasePadOp();
  for (const unsigned &flip : padBaseOp.getFlips()) {
    inTensor = inTensor.reverse(flip);
  }

  setOutTensor(BasePadOp::getOutIndex(), padGrow(inTensor, prog));
}

void PadInplaceOpx::grow(poplar::program::Sequence &prog) const {

  auto in0      = Opx::getInTensor(BasePadOp::getInIndex());
  auto inTensor = cloneNcopy(prog, in0);

  auto &&padBaseOp = getBasePadOp();
  for (const unsigned &flip : padBaseOp.getFlips()) {
    inTensor = inTensor.reverse(flip);
  }

  setOutTensor(BasePadOp::getOutIndex(), padGrow(inTensor, prog));
}

PadGradOpx::PadGradOpx(Op *op, Devicex *devicex) : SliceOpx(op, devicex) {
  verifyOp<PadGradOpx>(op, Onnx::GradOperators::PadGrad);
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
