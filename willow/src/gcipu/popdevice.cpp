#include <willow/error.hpp>
#include <willow/ir.hpp>
#include <willow/tensor.hpp>
#include <willow/gcipu/popdevice.hpp>

namespace willow {

PopDevice::PopDevice(const Ir *pir) : Device(pir) {
  poplar::IPUModel ipumodel;
  popDevice = ipumodel.createDevice();
  if (!popDevice.attach()) {
    throw error("failed to attach to popDevice");
  }
}

// go all the way to creating the engine
void PopDevice::prepare() {
  pGraph.reset(new poplar::Graph(popDevice));

  for (auto id : pir->tensors.getInitIds()){
    Tensor * initTensor = pir->tensors.get(id);
    Speck speck = initTensor->consumers.consensusSpeck();
  }

  // create poplar::Tensors etc.
  throw error("need to prepare poplar popDevice");
}



} // namespace willow

