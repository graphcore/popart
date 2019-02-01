#ifndef GUARD_NEURALNET_HALF_HPP
#define GUARD_NEURALNET_HALF_HPP
#include <cstdint>

namespace poponnx {

class Half {
public:
  [[noreturn]] Half operator+(const Half &);

private:
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-private-field"
  uint16_t data;
#pragma clang diagnostic pop
};

} // namespace poponnx

#endif
