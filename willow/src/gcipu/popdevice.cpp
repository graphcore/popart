#include <willow/error.hpp>
#include <willow/gcipu/popdevice.hpp>

namespace willow {

PopDevice::PopDevice(const Ir *pir) : Device(pir) {
  poplar::IPUModel ipumodel;
  popDevice = ipumodel.createDevice();
  if (!popDevice.attach()) {
    throw error("failed to attach to popDevice");
  }
}

void PopDevice::prepare() {
  pGraph.reset(new poplar::Graph(popDevice));

  // create poplar::Tensors etc.

  throw error("need to prepare poplar popDevice");
}

} // namespace willow

// implement poppopDevice here
