#ifndef GUARD_NEURALNET_HALF_HPP
#define GUARD_NEURALNET_HALF_HPP
#include <cstdint>

namespace poponnx {

class Half {

public:
  Half();
  ~Half();

  Half(const Half &other);
  Half(float f);

  Half &operator=(const Half &rhs);
  Half &operator=(const float rhs);

  bool operator==(const Half &rhs);

  Half operator+(const Half &);

  operator float() const;

private:
  uint16_t data;
};

using float16_t = Half;

std::ostream &operator<<(std::ostream &ss, const Half &v);

} // namespace poponnx

#endif
