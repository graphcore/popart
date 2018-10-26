#include <willow/device.hpp>
#include <willow/error.hpp>
#include <willow/popx/relux.hpp>
#include <willow/relu.hpp>

#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
#include <popnn/NonLinearity.hpp>
#pragma clang diagnostic pop // stop ignoring warnings

namespace willow {
namespace popx {

ReluOpx::ReluOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::RELU) {
    throw error("cannot create ReluOpx from " + op->op_type());
  }
}

void ReluOpx::grow() const {
  auto inName    = op_p->input.id(0);
  auto outName   = op_p->output.id(0);
  auto outTensor = graph().clone(get(inName));
  poplar::program::Copy copyProg(get(inName), outTensor);
  step().add(copyProg);
  popnn::nonLinearity(
      graph(), popnn::NonLinearityType::RELU, outTensor, step(), outName);

  insert(outName, outTensor);
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

} // namespace popx
} // namespace willow
