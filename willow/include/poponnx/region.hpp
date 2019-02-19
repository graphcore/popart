#ifndef GUARD_NEURALNET_REGIONIOMAP_HPP
#define GUARD_NEURALNET_REGIONIOMAP_HPP

#include <vector>
#include <poponnx/names.hpp>

// we currently only consider inplacing ops with 1 output. this can be
// generalised in the future if we decide it is necessary

namespace poponnx {
namespace view {

// a rectangular sub-region of a Shape
class Region {

public:
  Region(const std::vector<int64_t> &lower_,
         const std::vector<int64_t> &upper_);
  int64_t rank() const;
  int64_t nelms() const;
  bool isEmpty() const;
  Region intersect(const Region &rhs) const;
  void checks() const;
  static Region getEmpty(int64_t r);
  static Region getFull(const Shape &s);
  bool operator==(const Region &) const;
  bool operator!=(const Region &) const;
  const std::vector<int64_t> &getLower() const { return lower; }
  const std::vector<int64_t> &getUpper() const { return upper; }
  void append(std::ostream &ss) const;

private:
  std::vector<int64_t> lower;
  std::vector<int64_t> upper;
  // rank-0 tensors have no lower and upper bounds,
  // so it is not possible to determine if they are empty
  // by looking for equal lower and upper bounds
  bool isEmptyRank0{false};

  Region(const std::vector<int64_t> &lower_,
         const std::vector<int64_t> &upper_,
         bool isEmpty_r0_);
};

std::ostream &operator<<(std::ostream &stream, const Region &r);

} // namespace view
} // namespace poponnx

#endif
