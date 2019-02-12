#include <poponnx/error.hpp>
#include <poponnx/half.hpp>

#include <poplar/Target.hpp>

namespace poponnx {

static float halfToFloat(uint16_t f16) {
  // poplar::Target is not used in this function
  static auto dummyTarget = poplar::Target();
  float f;
  poplar::copyDeviceHalfToFloat(dummyTarget, &f16, &f, 1);
  return f;
}

static uint16_t floatToHalf(float f) {
  // poplar::Target is not used in this function
  static auto dummyTarget = poplar::Target();
  uint16_t f16;
  poplar::copyFloatToDeviceHalf(dummyTarget, &f, &f16, 1);
  return f16;
}

Half::Half() : data(0) {}

Half::~Half() {}

Half::Half(const Half &rhs) : data(rhs.data) {}

Half::Half(float f) : data(floatToHalf(f)) {}

Half &Half::operator=(const Half &rhs) {
  data = rhs.data;
  return *this;
}

Half &Half::operator=(const float rhs) {
  data = floatToHalf(rhs);
  return *this;
}

bool Half::operator==(const Half &rhs) { return this->data == rhs.data; }

Half Half::operator+(const Half &rhs) {
  return (static_cast<float>(*this) + static_cast<float>(rhs));
}

Half::operator float() const { return halfToFloat(this->data); }

std::ostream &operator<<(std::ostream &ss, const Half &v) {
  ss << static_cast<float>(v);
  return ss;
}

} // namespace poponnx
