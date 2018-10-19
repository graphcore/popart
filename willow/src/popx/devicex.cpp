#include <willow/error.hpp>
#include <willow/ir.hpp>
#include <willow/popx/addx.hpp>
#include <willow/popx/averagepoolx.hpp>
#include <willow/popx/convx.hpp>
#include <willow/popx/devicex.hpp>
#include <willow/popx/l1x.hpp>
#include <willow/popx/logsoftmaxx.hpp>
#include <willow/popx/opx.hpp>
#include <willow/popx/relux.hpp>
#include <willow/popx/squeezex.hpp>
#include <willow/tensor.hpp>

namespace willow {
namespace popx {

Devicex::Devicex(const Ir *pir) : willow::Device(pir) {
  poplar::IPUModel ipumodel;
  popDevice = ipumodel.createDevice();
  if (!popDevice.attach()) {
    throw error("failed to attach to popDevice");
  }
}

std::unique_ptr<Opx> Devicex::createOpx(Op *op) {
  switch (op->opType) {
  case OpType::ADD: {
    return std::unique_ptr<Opx>(new AddOpx(op));
  }
  case OpType::CONV: {
    return std::unique_ptr<Opx>(new ConvOpx(op));
  }
  case OpType::RELU: {
    return std::unique_ptr<Opx>(new ReluOpx(op));
  }
  case OpType::AVERAGEPOOL: {
    return std::unique_ptr<Opx>(new AveragePoolOpx(op));
  }
  case OpType::L1GRAD: {
    return std::unique_ptr<Opx>(new L1GradOpx(op));
  }
  case OpType::L1: {
    return std::unique_ptr<Opx>(new L1Opx(op));
  }
  case OpType::LOGSOFTMAX: {
    return std::unique_ptr<Opx>(new LogSoftmaxOpx(op));
  }
  case OpType::SQUEEZE: {
    return std::unique_ptr<Opx>(new SqueezeOpx(op));
  }

  case OpType::ADDGRAD:
  case OpType::AVERAGEPOOLGRAD:
  case OpType::CONSTANT:
  case OpType::CONVDATAGRAD:
  case OpType::CONVWEIGHTSGRAD:
  case OpType::LOGSOFTMAXGRAD:
  case OpType::NLL:
  case OpType::NLLGRAD:
  case OpType::PAD:
  case OpType::RELUGRAD:
  case OpType::SQUEEZEGRAD:
  case OpType::SUM:
  case OpType::VARUPDATE:
    throw error("No get pop op for " + op->op_type());
  }
}

Opx *Devicex::getOpx(OpId id) { return opxs.at(id).get(); }

// Design decision : leave the option for a Tensor to be
// created based on complex global criteria open.
poplar::Tensor Devicex::createPopTensor(Tensor *tensor) {
  // if there is just one consumer, does it know
  // how to create a poplar::Tensor?
  if (tensor->consumers.getTotal() == 1) {
    Op *op        = tensor->consumers.getOps()[0];
    auto conIndex = op->input.indices(tensor)[0];
    auto conId    = op->id;
    auto popOp    = getOpx(conId);
    if (popOp->canCreateInput(conIndex)) {
      return popOp->createInput(conIndex);
    }
  }
  throw error("beflumoxed in create tensor");
}

// go all the way to creating the engine
void Devicex::prepare() {
  pGraph.reset(new poplar::Graph(popDevice));

  // create a Opx for every Op
  for (Op *op : pir->getTopologicallySorted()) {
    opxs[op->id] = createOpx(op);
  }

  // create poplar::Tensor for each of the initializers
  for (auto id : pir->tensors.getInitIds()) {
    Tensor *tensor          = pir->tensors.get(id);
    pop_tensors[tensor->id] = createPopTensor(tensor);
  }

  // create poplar::Tensors etc.
  throw error("need to prepare poplar popDevice");
}

} // namespace popx
} // namespace willow
