#ifndef GUARD_NEURALNET_SCATTERX_HPP
#define GUARD_NEURALNET_SCATTERX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {
namespace popx {

class ScatterOpx : public Opx {
public:
  ScatterOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  int64_t axis;
};

class ScatterDataGradOpx : public Opx {
public:
  ScatterDataGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  int64_t axis;
};

class ScatterUpdateGradOpx : public Opx {
public:
  ScatterUpdateGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  int64_t axis;
};

} // namespace popx
} // namespace popart

#endif
