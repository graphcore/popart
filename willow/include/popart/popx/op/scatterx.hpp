// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SCATTERX_HPP
#define GUARD_NEURALNET_SCATTERX_HPP

#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {
namespace popx {

class ScatterOpx : public PopOpx {
public:
  ScatterOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  int64_t axis;
};

class ScatterDataGradOpx : public PopOpx {
public:
  ScatterDataGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  int64_t axis;
};

class ScatterUpdateGradOpx : public PopOpx {
public:
  ScatterUpdateGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  int64_t axis;
};

} // namespace popx
} // namespace popart

#endif
