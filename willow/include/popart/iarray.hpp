#ifndef GUARD_NEURALNET_IARRAY_HPP
#define GUARD_NEURALNET_IARRAY_HPP
#include <popart/tensorinfo.hpp>

namespace popart {

class IArray {
public:
  virtual ~IArray() {}
  virtual void *data()                 = 0;
  virtual DataType dataType() const    = 0;
  virtual std::size_t rank() const     = 0;
  virtual int64_t dim(int index) const = 0;
  virtual std::size_t nelms() const    = 0;
};

} // namespace popart

#endif
