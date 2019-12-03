#ifndef GUARD_NEURALNET_REGIONIOMAP_HPP
#define GUARD_NEURALNET_REGIONIOMAP_HPP

#include <memory>
#include <set>
#include <vector>
#include <popart/names.hpp>

// we currently only consider inplacing ops with 1 output. this can be
// generalised in the future if we decide it is necessary

namespace popart {
namespace view {

Regions mergeRegions(Regions regions);

// a rectangular sub-region of a Shape
class Region {

public:
  Region(const std::vector<int64_t> &lower_,
         const std::vector<int64_t> &upper_);
  int64_t rank() const;
  int64_t nelms() const;
  bool isEmpty() const;
  Region intersect(const Region &rhs) const;
  Region transpose(const Shape shape) const;
  Regions sub(const Regions &rhs, bool include_empty = false) const;
  Regions sub(const Region &rhs, bool include_empty = false) const;
  Regions add(const Region &rhs) const;
  Regions cut(const std::vector<std::set<int64_t>> &cuts,
              bool include_empty = false) const;
  Regions reshape(Region fullInRegion, Region fullOutRegion) const;
  std::pair<int64_t, Region> merge(const Region &rhs) const;
  bool contains(const std::vector<int64_t> &index) const;
  bool contains(const Region &rhs) const;
  int64_t flatIndex(const std::vector<int64_t> &index) const;
  std::vector<int64_t> dimIndex(int64_t index) const;
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
} // namespace popart

#endif
