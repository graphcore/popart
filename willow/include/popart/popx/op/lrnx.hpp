#ifndef GUARD_NEURALNET_LRNX_HPP
#define GUARD_NEURALNET_LRNX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {

namespace popx {

class LRNOpx : public Opx {
public:
  LRNOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class LRNGradOpx : public Opx {
public:
  LRNGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
