#include <willow/device.hpp>

namespace willow {

Device::~Device() = default;

Device::Device(const Graph *g) : graph(g) {}

} // namespace willow
