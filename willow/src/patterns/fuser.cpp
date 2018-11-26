#include <poponnx/error.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/patterns/fuser.hpp>
#include <poponnx/pbwrap.hpp>
#include <poponnx/tensor.hpp>

namespace willow {

void Fuser::apply(Op *op) const {
  Ir *pir = op->pir;

  Op *op0      = op;
  Tensor *out0 = op0->output.tensor(0);
  Op *op1      = out0->consumers.getOps()[0];
  Tensor *out1 = op1->output.tensor(0);

  // create the replacement op01, connect it to
  // - the inputs if op0
  // - the output of op1
  OpId id01 = moveMergedIntoIr(op);
  Op *op01  = pir->getOp(id01);

  // wire-up the inputs
  pir->connectInputsFromInputMapWrapper(
      InputMapWrapper(op0->input.tensorIdMap()), id01);
  for (auto index_tensor : op0->input.tensorMap()) {
    Tensor *in0 = index_tensor.second;
    in0->consumers.decrement(op0);
  }

  // we can't use connectOutputs, as that expects
  // that the output Tensor doesn't exist and must
  // be created. We rewire outputs manually:
  op01->output.insert(0, out1);
  out1->resetProducer(op01);

  // remove the tensor and nodes
  pir->getTensors().remove(out0->id);
  pir->eraseOp(op0->id);
  pir->eraseOp(op1->id);
}

bool Fuser::matches(Op *op0) const {
  if (op0->opType == get0()) {
    const Tensor *ten_d = op0->output.tensor(0);
    // Consumed just once? Should be the case
    if (ten_d->consumers.getTotal() == 1) {
      Op *op1 = ten_d->consumers.getOps()[0];
      if (op1->opType == get1()) {
        return true;
      }
    }
  }
  return false;
}

std::vector<const Tensor *> Fuser::touches(Op *op) const {
  return {op->output.tensor(0)};
}

} // namespace willow
