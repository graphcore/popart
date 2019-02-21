#include <poponnx/error.hpp>
#include <poponnx/region.hpp>
#include <poponnx/util.hpp>

#include <boost/range/algorithm.hpp>

namespace poponnx {
namespace view {

bool Region::operator==(const Region &r) const {
  return lower == r.lower && upper == r.upper && isEmptyRank0 == r.isEmptyRank0;
}

bool Region::operator!=(const Region &r) const { return !(r == *this); }

Region::Region(const std::vector<int64_t> &l, const std::vector<int64_t> &u)
    : Region(l, u, false) {}

Region::Region(const std::vector<int64_t> &l,
               const std::vector<int64_t> &u,
               bool er0)
    : lower(l), upper(u), isEmptyRank0(er0) {
  checks();
}

void Region::checks() const {
  if (lower.size() != upper.size()) {
    throw error("lower of size {}, upper of size {}, in Region::checks",
                lower.size(),
                upper.size());
  }

  for (int64_t i = 0; i < rank(); ++i) {
    if (lower[i] > upper[i]) {
      throw error("lower bound {}, is greater than upper bound, in "
                  "Region::checks for dimension {}",
                  lower[i],
                  upper[i],
                  i);
    }
  }

  if (isEmptyRank0 && lower.size() != 0) {
    throw error("ILE: cannot be `empty-of-rank-0' if it is not rank 0!");
  }
}

Region Region::getEmpty(int64_t r) {
  // One possible empty region
  return Region(LowBounds(r, 0), UppBounds(r, 0), r == 0 ? true : false);
}

Region Region::getFull(const Shape &s) {
  // Use the Shape as the UppBounds
  return Region(LowBounds(s.size(), 0), s, false);
}

int64_t Region::rank() const { return lower.size(); }

int64_t Region::nelms() const {
  if (isEmptyRank0) {
    return 0;
  }

  int64_t n = 1;
  for (int64_t i = 0; i < rank(); ++i) {
    n *= (upper[i] - lower[i]);
  }
  return n;
}

bool Region::isEmpty() const { return nelms() == 0; }

Region Region::intersect(const Region &rhs) const {
  if (*this == rhs) {
    return rhs;
  }

  if (rank() != rhs.rank()) {
    throw error("Regions are of different rank in intersect");
  }
  if (rhs.isEmpty() || isEmpty()) {
    return getEmpty(rhs.rank());
  }
  Region result(lower, upper);

  // Resolve templates and overload set
  const auto min = [](int64_t a, int64_t b) { return std::min(a, b); };
  const auto max = [](int64_t a, int64_t b) { return std::max(a, b); };

  boost::transform(lower, rhs.lower, result.lower.begin(), max);
  boost::transform(upper, rhs.upper, result.upper.begin(), min);
  boost::transform(result.lower, result.upper, result.lower.begin(), min);
  boost::transform(result.lower, result.upper, result.upper.begin(), max);

  return result;
}

void Region::append(std::ostream &ss) const {
  ss << "L:";
  appendSequence(ss, lower);
  ss << " U:";
  appendSequence(ss, upper);
}

std::ostream &operator<<(std::ostream &stream, const Region &r) {
  r.append(stream);
  return stream;
}

} // namespace view
} // namespace poponnx
