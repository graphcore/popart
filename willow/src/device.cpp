#include <poponnx/device.hpp>

namespace willow {

Device::~Device() = default;

Device::Device(const Ir *g) : pir(g) {}

const Ir *Device::ir() const { return pir; }

} // namespace willow
