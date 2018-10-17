#include <willow/error.hpp>
#include <willow/gcipu/popdevice.hpp>

namespace willow {

PopDevice::PopDevice(const Graph *g) : Device(g) {
  std::cout << "Create pop device (from string?)" << std::endl;
}

void PopDevice::prepare() {
throw error("need to prepare poplar device");
}

} // namespace willow

// implement popdevice here
