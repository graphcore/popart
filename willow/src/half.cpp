// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <cstdint>
#include <iosfwd>
#include <poplar/Target.hpp>
#include <popart/half.hpp>

namespace popart {

float halfToFloat(uint16_t f16) {
  // poplar::Target is not used in this function
  static auto dummyTarget = poplar::Target();
  float f;
  poplar::copyDeviceHalfToFloat(dummyTarget, &f16, &f, 1);
  return f;
}

uint16_t floatToHalf(float f) {
  // poplar::Target is not used in this function
  static auto dummyTarget = poplar::Target();
  uint16_t f16;
  poplar::copyFloatToDeviceHalf(dummyTarget, &f, &f16, 1);
  return f16;
}

std::ostream &operator<<(std::ostream &ss, const Half &v) {
  ss << static_cast<float>(v);
  return ss;
}

} // namespace popart
