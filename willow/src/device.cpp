#include <willow/device.hpp>

namespace willow {

Device::~Device() = default;

Device::Device(const Ir *g) : pir(g) {}

} // namespace willow
