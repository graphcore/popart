#include <willow/error.hpp>
#include <willow/ir.hpp>
#include <willow/tensor.hpp>
#include <willow/popx/popdevice.hpp>
#include <willow/popx/popop.hpp>

namespace willow {

PopDevice::PopDevice(const Ir *pir) : Device(pir) {
  poplar::IPUModel ipumodel;
  popDevice = ipumodel.createDevice();
  if (!popDevice.attach()) {
    throw error("failed to attach to popDevice");
  }
}

std::unique_ptr<PopOp> PopDevice::createPopOp(const Op *op) {
  switch (op->opType) {
  case OpType::ADD:
  case OpType::ADDGRAD:
  case OpType::AVERAGEPOOL:
  case OpType::AVERAGEPOOLGRAD:
  case OpType::CONSTANT:
  case OpType::CONV:
  case OpType::CONVDATAGRAD:
  case OpType::CONVWEIGHTSGRAD:
  case OpType::L1:
  case OpType::L1GRAD:
  case OpType::LOGSOFTMAX:
  case OpType::LOGSOFTMAXGRAD:
  case OpType::NLL:
  case OpType::NLLGRAD:
  case OpType::PAD:
  case OpType::RELU:
  case OpType::RELUGRAD:
  case OpType::SQUEEZE:
  case OpType::SQUEEZEGRAD:
  case OpType::SUM:
  case OpType::VARUPDATE:
    throw error("No get pop op for " + op->op_type());
  }
}

PopOp * PopDevice::getPopOp(OpId id){
 return pop_ops.at(id).get();
}

// Design decision : leave the option for a Tensor to be
// created based on complex global criteria open.
poplar::Tensor PopDevice::createPopTensor(Tensor *tensor) {
  // if there is just one consumer, does it know
  // how to create a poplar::Tensor?
  if (tensor->consumers.getTotal() == 1) {
    Op *op        = tensor->consumers.getOps()[0];
    auto conIndex = op->input.indices(tensor)[0];
    auto conId    = op->id;
    auto popOp    = getPopOp(conId);
    if (popOp->canCreateInput(conIndex)) {
      return popOp->createInput(conIndex);
    }
  }
  throw error("beflumoxed in create tensor");
}

// go all the way to creating the engine
void PopDevice::prepare() {
  pGraph.reset(new poplar::Graph(popDevice));

  // create a PopOp for every Op
  for (Op * op : pir->getTopologicallySorted()){
    pop_ops[op->id] = createPopOp(op);
  }

  // create poplar::Tensor for each of the initializers
  for (auto id : pir->tensors.getInitIds()) {
    Tensor *tensor = pir->tensors.get(id);
    pop_tensors[tensor->id] = createPopTensor(tensor);
  }

  // create poplar::Tensors etc.
  throw error("need to prepare poplar popDevice");
}



} // namespace willow


   
