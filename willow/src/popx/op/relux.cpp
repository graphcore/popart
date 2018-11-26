#include <poponnx/device.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/relu.hpp>
#include <poponnx/popx/op/relux.hpp>

#include <popnn/NonLinearity.hpp>

namespace willow {
namespace popx {

ReluOpx::ReluOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::RELU) {
    throw error("cannot create ReluOpx from " + op->op_type());
  }
}

ReluInplaceOpx::ReluInplaceOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::RELUINPLACE) {
    throw error("cannot create ReluInplaceOpx from " + op->op_type());
  }
}

void ReluOpx::grow(poplar::program::Sequence &prog) const {

  // There is only an in-place poplibs Relu. We therefore clone first,
  auto outTensor = cloneNcopy(prog, inId(0));

  // and apply the inplace relu.
  popnn::nonLinearityInPlace(
      graph(), popnn::NonLinearityType::RELU, outTensor, prog, outId(0));

  insert(outId(0), outTensor);
}

void ReluInplaceOpx::grow(poplar::program::Sequence &prog) const {
  // apply the inplace relu,
  popnn::nonLinearityInPlace(
      graph(), popnn::NonLinearityType::RELU, get(inId(0)), prog, inId(0));
}

ReluOp *ReluOpx::getReluOp() const { return dynamic_cast<ReluOp *>(op_p); }

ReluInplaceOp *ReluInplaceOpx::getReluInplaceOp() const {
  return dynamic_cast<ReluInplaceOp *>(op_p);
}

ReluGradOpx::ReluGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::RELUGRAD) {
    throw error("cannot create ReluGradOpx from " + op->op_type());
  }
}

ReluGradOp *ReluGradOpx::getReluGradOp() const {
  return dynamic_cast<ReluGradOp *>(op_p);
}

void ReluGradOpx::grow(poplar::program::Sequence &prog) const {

  ReluGradOp *rgop = getReluGradOp();

  auto outTensor = popnn::nonLinearityInputGradient(
      graph(),                           // graph,
      popnn::NonLinearityType::RELU,     // nonLinearityType,
      get(inId(rgop->getReludIn())),     //  out,
      get(inId(rgop->getGradReludIn())), //  outGradient,
      prog,                              // prog,
      idStr()                            // debugPrefix
  );

  insert(op_p->output.id(0), outTensor);
}

} // namespace popx
} // namespace willow
