#ifndef GUARD_NEURALNET_REGIONIOMAP_HPP
#define GUARD_NEURALNET_REGIONIOMAP_HPP

#include <poponnx/names.hpp>

// we currently only consider inplacing ops with 1 output. this can be
// generalised in the future if we decide it is necessary

namespace poponnx {

// a class describing a subset of a tensor obtained through slices, concats,
// unions, differences, etc. Currently, we assume that ALL operations on the
// tensor have no effect, so that this Region always a superset of the true
// region.
class Region {
public:
  Region(bool x_) : x(x_) {}
  bool x;
};

class RegionIO {
public:
  // TODO how big do we expect these Region objects to be in the future?
  // Might consider a move constructor
  RegionIO(const Region &i, const Region &o) : region_in(i), region_out(o) {}
  const Region &in() { return region_in; }
  const Region &out() { return region_out; }

private:
  Region region_in;
  Region region_out;
};

class RegionIOMap {

public:
  RegionIOMap(std::map<InIndex, RegionIO> &&m_) : m(m_) {}
  const Region &in(InIndex i) { return m.at(i).in(); }
  const Region &out(InIndex i) { return m.at(i).out(); }

private:
  std::map<InIndex, RegionIO> m;
};

} // namespace poponnx

#endif
