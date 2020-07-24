// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_HALF_HPP
#define GUARD_NEURALNET_HALF_HPP
#include <cstdint>

namespace popart {

// We define these some functions in this header itself because they are one
// of the most prominent popart functions in terms of number of calls.
// Optimizing their call time may impact applications.

extern float halfToFloat(uint16_t f16);
extern uint16_t floatToHalf(float f);

class Half {

public:
  Half() : data(0) {}
  ~Half() {}

  Half(const Half &rhs) : data(rhs.data) {}
  Half(float f) : data(floatToHalf(f)) {}

  // Catch all arithmetic types and handle the explicit cast to float
  template <
      typename T,
      typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
  explicit Half(T f) : Half(static_cast<float>(f)) {}

  Half &operator=(const Half &rhs) {
    data = rhs.data;
    return *this;
  }

  Half &operator=(const float rhs) {
    data = floatToHalf(rhs);
    return *this;
  }

  bool operator==(const Half &rhs) { return this->data == rhs.data; }

  Half operator+(const Half &rhs) {
    return (static_cast<float>(*this) + static_cast<float>(rhs));
  }

  Half operator/(const Half &rhs) {
    return (static_cast<float>(*this) / static_cast<float>(rhs));
  }

  operator float() const { return halfToFloat(this->data); }

private:
  uint16_t data;
};

using float16_t = Half;

std::ostream &operator<<(std::ostream &ss, const Half &v);

} // namespace popart

#endif
