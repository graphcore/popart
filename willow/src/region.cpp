#include <utility>

#include <popart/error.hpp>
#include <popart/region.hpp>
#include <popart/util.hpp>

#include <boost/range/algorithm.hpp>

namespace popart {
namespace view {

// Merge in approx. O(rank * 2 * n log^2 n)
Regions mergeRegions(Regions regions) {
  size_t last_size = 0;
  if (regions.size() > 0) {
    Regions subRegions;

    while (regions.size() != last_size || subRegions.size() > 0) {

      regions.insert(regions.end(), subRegions.begin(), subRegions.end());
      subRegions.clear();

      auto rank = regions.front().rank();

      for (int64_t d = 0; d < rank; ++d) {
        std::sort(regions.begin(),
                  regions.end(),
                  [d](const view::Region &a, const view::Region &b) -> bool {
                    return a.getLower()[d] < b.getLower()[d];
                  });

        std::vector<bool> erase(regions.size(), false);

        for (int64_t i = 0; i < regions.size(); ++i) {
          if (regions[i].isEmpty())
            erase[i] = true;
          if (erase[i])
            continue;
          for (int64_t j = i + 1; j < regions.size(); ++j) {
            if (erase[j])
              continue;
            if (regions[i].getUpper()[d] < regions[j].getLower()[d]) {
              break;
            } else {
              // Try merge
              auto merged = regions[i].merge(regions[j]);
              if (merged.first == d) {
                regions[i] = merged.second;
                erase[j]   = true;
              } else {
                // Try subtraction
                auto subs = regions[j].sub(regions[i]);
                if (subs.size() > 0 && subs.front() != regions[j]) {
                  subRegions.insert(subRegions.end(), subs.begin(), subs.end());
                  erase[j] = true;
                }
              }
            }
          }
        }

        regions.erase(std::remove_if(regions.begin(),
                                     regions.end(),
                                     [&erase, &regions](Region const &i) {
                                       return erase.at(&i - regions.data());
                                     }),
                      regions.end());
      }
      last_size = regions.size();
    }
  }
  return regions;
}

bool Region::operator==(const Region &r) const {
  // is this correct for empty regions?
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
      throw error("lower bound {}, is greater than upper bound {}, in "
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
    std::ostringstream oss;
    oss << "Internal Logic Error: Regions of different rank in intersect. ";
    oss << "\n     First Region " << *this;
    oss << "\n     Second Region " << rhs;
    throw error(oss.str());
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

Regions Region::sub(const Region &rhs, bool include_empty) const {
  Regions result;

  if (*this == rhs) {
    return result;
  }

  if (rank() != rhs.rank()) {
    throw error(
        "Regions are of different rank ({} vs. {}) in sub", rank(), rhs.rank());
  }
  if (rhs.isEmpty()) {
    return {*this};
  }

  std::vector<std::set<int64_t>> cuts(rank());

  for (int64_t d = 0; d < rank(); ++d) {
    cuts[d].insert(rhs.getLower()[d]);
    cuts[d].insert(rhs.getUpper()[d]);
  }

  result = cut(cuts, include_empty);

  Regions filteredResults;
  for (auto r : result) {
    if (rhs.intersect(r) != r) {
      // r not fully contained in rhs
      filteredResults.push_back(r);
    }
  }
  filteredResults = mergeRegions(filteredResults);

  return filteredResults;
}

Regions Region::sub(const Regions &rhs, bool /* include_empty */) const {
  Regions rqueue;
  Regions wqueue;
  rqueue.push_back(*this);
  for (auto &r0 : rhs) {
    for (auto &r1 : rqueue) {
      for (auto &r2 : r1.sub(r0)) {
        wqueue.push_back(r2);
      }
    }
    rqueue = wqueue;
    wqueue.clear();
  }
  rqueue = mergeRegions(rqueue);
  return rqueue;
}

Regions Region::cut(const std::vector<std::set<int64_t>> &cuts,
                    bool include_empty) const {
  Regions rqueue(1, *this);
  Regions wqueue;

  for (int64_t i = 0; i < cuts.size(); ++i) {
    for (int64_t cut : cuts[i]) {
      while (rqueue.size() > 0) {
        Region r0 = rqueue.back();
        rqueue.pop_back();
        if (cut > r0.getLower()[i] && cut < r0.getUpper()[i]) {
          std::vector<int64_t> l1 = r0.getLower();
          std::vector<int64_t> u1 = r0.getUpper();
          std::vector<int64_t> l2 = r0.getLower();
          std::vector<int64_t> u2 = r0.getUpper();
          u1[i]                   = cut;
          l2[i]                   = cut;
          Region r1(l1, u1);
          Region r2(l2, u2);
          if (r1.nelms() > 0 || include_empty)
            wqueue.push_back(r1);
          if (r2.nelms() > 0 || include_empty)
            wqueue.push_back(r2);
        } else {
          if (r0.nelms() > 0 || include_empty)
            wqueue.push_back(r0);
        }
      }
      rqueue = wqueue;
      wqueue.clear();
    }
  }
  return rqueue;
}

bool Region::contains(const std::vector<int64_t> &index) const {
  bool contained = true;
  for (int64_t i = 0; i < index.size(); ++i) {
    contained &= (lower[i] <= index[i] && index[i] < upper[i]);
  }
  return index.size() == rank() && contained;
}

bool Region::contains(const Region &rhs) const {
  auto rhsl = rhs.getLower();
  auto rhsu = rhs.getUpper();

  for (auto &x : rhsu)
    x--;

  return rank() == rhs.rank() && contains(rhsl) && contains(rhsu);
}

int64_t Region::flatIndex(const std::vector<int64_t> &index) const {
  int64_t flat = 0;
  for (int64_t d = 0; d < rank(); ++d) {
    flat += index[d];
    if (d < rank() - 1)
      flat *= (upper[d + 1] - lower[d + 1]);
  }
  return flat;
}

std::vector<int64_t> Region::dimIndex(int64_t index) const {
  std::vector<int64_t> dim(rank());
  for (int64_t d = rank() - 1; d >= 0; --d) {
    int64_t size = (upper[d] - lower[d]);
    dim[d]       = index % size;
    index        = index / size;
  }
  return dim;
}

// When fullInRegion is reshaped to fullOutRegion, this region within
// fullInRegion maps to which regions of fullOutRegion
Regions Region::reshape(Region fullInRegion, Region fullOutRegion) const {
  Regions regions;

  if (fullInRegion.nelms() != fullOutRegion.nelms()) {
    throw error("Regions are of different element size");
  }

  if (rank() != fullInRegion.rank()) {
    throw error("This region and the in-region are of different rank");
  }

  if (*this == fullInRegion) {
    // If in == fullInRegion, then out == fullOutRegion
    return {fullOutRegion};
  }

  int64_t step_in  = 1;
  int64_t step_out = 1;
  int64_t cut_dim;
  for (cut_dim = 1;
       cut_dim <= std::min(fullInRegion.rank(), fullOutRegion.rank());
       ++cut_dim) {
    step_in *= fullInRegion.getUpper()[fullInRegion.rank() - cut_dim] -
               fullInRegion.getLower()[fullInRegion.rank() - cut_dim];
    step_out *= fullOutRegion.getUpper()[fullOutRegion.rank() - cut_dim] -
                fullOutRegion.getLower()[fullOutRegion.rank() - cut_dim];
    if (fullInRegion.getLower()[fullInRegion.rank() - cut_dim] !=
            fullOutRegion.getLower()[fullOutRegion.rank() - cut_dim] ||
        fullInRegion.getUpper()[fullInRegion.rank() - cut_dim] !=
            fullOutRegion.getUpper()[fullOutRegion.rank() - cut_dim])
      break;
  }

  // Calculate where the out region cuts the in region
  std::vector<int64_t> cut_points;
  cut_points.reserve(fullInRegion.nelms() / step_in +
                     fullOutRegion.nelms() / step_out);

  for (int64_t i = step_in; i < fullInRegion.nelms(); i += step_in) {
    cut_points.push_back(i);
  }
  for (int64_t i = step_out; i < fullOutRegion.nelms(); i += step_out) {
    cut_points.push_back(i);
  }

  std::vector<std::set<int64_t>> cuts(fullInRegion.rank());

  for (auto cut_point : cut_points) {
    std::vector<int64_t> dim = fullInRegion.dimIndex(cut_point);
    for (int64_t d = 0; d < fullInRegion.rank(); ++d) {
      cuts[d].insert(dim[d]);
    }
  }

  // Cut the in region
  Regions cutRegions = cut(cuts);

  // Intersect this with cutRegions
  for (auto region : cutRegions) {
    if (!region.isEmpty()) {
      auto l = region.getLower();
      auto u = region.getUpper();
      for (auto &x : u)
        x--;
      auto l_flat = fullInRegion.flatIndex(l);
      auto u_flat = fullInRegion.flatIndex(u);
      l           = fullOutRegion.dimIndex(l_flat);
      u           = fullOutRegion.dimIndex(u_flat);
      for (auto &x : u)
        x++;
      regions.emplace_back(l, u);
    }
  }
  regions = mergeRegions(regions);

  bool verifyArea = false;
  if (verifyArea) {
    int64_t reshapeAreas{0};
    for (auto x : regions) {
      reshapeAreas += x.nelms();
    }
    if (reshapeAreas != nelms()) {
      throw error("Internal Logic Error: number of elements in output of "
                  "Region::reshape differents from the number of elements of "
                  "the input ({} != {})",
                  reshapeAreas,
                  nelms());
    }
  }

  return regions;
}

std::pair<int64_t, Region> Region::merge(const Region &rhs) const {
  bool can_merge    = true;
  int64_t merge_dim = -1;

  if (contains(rhs)) {
    return std::make_pair<int64_t, Region>(0, Region(getLower(), getUpper()));
  }
  if (rhs.contains(*this)) {
    return std::make_pair<int64_t, Region>(
        0, Region(rhs.getLower(), rhs.getUpper()));
  }

  for (int64_t d = 0; d < rank(); ++d) {
    if (lower[d] != rhs.lower[d] || upper[d] != rhs.upper[d]) {
      if (merge_dim > -1) {
        can_merge = false;
        break;
      }
      if (!(lower[d] <= rhs.upper[d] && rhs.lower[d] <= upper[d])) {
        can_merge = false;
        break;
      }
      merge_dim = d;
    }
  }

  if (can_merge) {
    std::vector<int64_t> newLower = lower;
    std::vector<int64_t> newUpper = upper;
    newLower[merge_dim] = std::min(lower[merge_dim], rhs.lower[merge_dim]);
    newUpper[merge_dim] = std::max(upper[merge_dim], rhs.upper[merge_dim]);
    return {merge_dim, Region(newLower, newUpper)};
  } else {
    return {-1, Region::getEmpty(rank())};
  }
}

void Region::append(std::ostream &ss) const {
  ss << "L:";
  appendSequence(ss, lower);
  ss << " U:";
  appendSequence(ss, upper);
}

Region Region::transpose(const Shape perm) const {
  std::vector<int64_t> l(perm.size());
  std::vector<int64_t> u(perm.size());

  for (int64_t i = 0; i < perm.size(); ++i) {
    l[i] = lower[perm[i]];
    u[i] = upper[perm[i]];
  }
  return Region(l, u);
}

std::ostream &operator<<(std::ostream &stream, const Region &r) {
  r.append(stream);
  return stream;
}

} // namespace view
} // namespace popart
