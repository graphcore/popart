#include <poponnx/device.hpp>
#include <poponnx/error.hpp>
#include <poponnx/popx/relux.hpp>
#include <poponnx/relu.hpp>

#include <popnn/NonLinearity.hpp>

namespace willow {
namespace popx {

ReluOpx::ReluOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::RELU) {
    throw error("cannot create ReluOpx from " + op->op_type());
  }
}

void ReluOpx::grow() const {

  // There is only an in-place poplibs Relu. We therefore clone first,
  auto outTensor = cloneNcopy(inId(0));

  // and apply the inplace relu.
  popnn::nonLinearityInPlace(
      graph(), popnn::NonLinearityType::RELU, outTensor, step(), outId(0));

  insert(outId(0), outTensor);
}

ReluOp *ReluOpx::getReluOp() const { return dynamic_cast<ReluOp *>(op_p); }

ReluGradOpx::ReluGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::RELUGRAD) {
    throw error("cannot create ReluGradOpx from " + op->op_type());
  }
}

ReluGradOp *ReluGradOpx::getReluGradOp() const {
  return dynamic_cast<ReluGradOp *>(op_p);
}

void ReluGradOpx::grow() const {

  ReluGradOp *rgop = getReluGradOp();

  auto outTensor = popnn::nonLinearityInputGradient(
      graph(),                           // graph,
      popnn::NonLinearityType::RELU,     // nonLinearityType,
      get(inId(rgop->getReludIn())),     //  out,
      get(inId(rgop->getGradReludIn())), //  outGradient,
      step(),                            // prog,
      idStr()                            // debugPrefix
  );

  insert(op_p->output.id(0), outTensor);
}

} // namespace popx
} // namespace willow
