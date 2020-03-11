#include <popops/Pad.hpp>
#include <popart/error.hpp>
#include <popart/op/pad.hpp>
#include <popart/popx/op/padgradx.hpp>
#include <popart/popx/op/padx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/util.hpp>

namespace popart {
namespace popx {

PadOpx::PadOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<PadOp>(op);
}

namespace {
poplar::Tensor pad_grow(const poplar::Tensor &inTensor,
                        poplar::Graph &graph,
                        const BasePadOp &pad_base_op) {
  auto &&pads    = pad_base_op.getPads();
  auto pad_value = pad_base_op.getPadValue();

  auto t_rank = inTensor.rank();
  auto &&mode = pad_base_op.getMode();

  std::vector<std::ptrdiff_t> lower_padding;
  std::vector<std::ptrdiff_t> upper_padding;
  for (int i = 0; i < t_rank; i++) {
    lower_padding.push_back(pads[i]);
    upper_padding.push_back(pads[t_rank + i]);
  }

  poplar::Tensor out_tensor;

  const auto logLevel = logging::Level::Trace;
  if (logging::shouldLog(logging::Module::devicex, logLevel)) {
    std::ostringstream oss;
    oss << "In pad_grow with mode = " << mode << ", pad_value = " << pad_value
        << ", lower_padding = ";
    appendSequence(oss, lower_padding);
    oss << ", upper_padding = ";
    appendSequence(oss, upper_padding);
    logging::log(logging::Module::devicex, logLevel, oss.str());
  }

  if (mode == "constant") {
    out_tensor =
        popops::pad(graph, inTensor, lower_padding, upper_padding, pad_value);
  } else if (mode == "edge") {
    out_tensor = popops::pad(
        inTensor, lower_padding, upper_padding, popops::padding::Type::EDGE);
  } else if (mode == "reflect") {
    out_tensor = popops::pad(
        inTensor, lower_padding, upper_padding, popops::padding::Type::REFLECT);
  } else {
    throw error("Bad mode type `{}' passed to pad op", mode);
  }

  return out_tensor;
}
} // namespace

void PadOpx::grow(poplar::program::Sequence &prog) const {
  auto out_tensor =
      pad_grow(getInTensor(BasePadOp::getInIndex()), graph(), *getPadOp());

  // As this is not an inplace op, and everything til here has just been view
  // changing, we clone and copy. Indeed we have a test showing that without
  // this cloneNcopy, the result is incorrect.

  out_tensor = cloneNcopy(prog, out_tensor);
  setOutTensor(BasePadOp::getOutIndex(), out_tensor);
}

void PadInplaceOpx::grow(poplar::program::Sequence &) const {

  setOutTensor(BasePadOp::getOutIndex(),
               pad_grow(getInTensor(BasePadOp::getInIndex()),
                        graph(),
                        *getPadInplaceOp()));
}

PadOp *PadOpx::getPadOp() const { return dynamic_cast<PadOp *>(op_p); }

PadInplaceOp *PadInplaceOpx::getPadInplaceOp() const {
  return dynamic_cast<PadInplaceOp *>(op_p);
}

PadGradOpx::PadGradOpx(Op *op, Devicex *devicex) : SliceOpx(op, devicex) {
  verifyOp<PadGradOpx>(op, Onnx::GradOperators::PadGrad);
}

PadInplaceOpx::PadInplaceOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<PadInplaceOp>(op, Onnx::CustomOperators::PadInplace);
}

namespace {
OpxCreator<PadOpx> padOpxCreator({Onnx::Operators::Pad_2,
                                  Onnx::Operators::Pad_11});
OpxCreator<PadGradOpx> padGradOpxCreator(Onnx::GradOperators::PadGrad);
OpxCreator<PadInplaceOpx>
    padxInplaceOpxCreator(Onnx::CustomOperators::PadInplace);
} // namespace

} // namespace popx
} // namespace popart
