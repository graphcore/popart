// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TOPKX_HPP
#define GUARD_NEURALNET_TOPKX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/basesortx.hpp>

namespace popart {

namespace popx {

class TopKOpx : public BaseSortOpx {
public:
  TopKOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  unsigned K;
};

class TopKGradOpx : public PopOpx {
public:
  TopKGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  // The info of the output of this Op
  const std::vector<size_t> &getGradOutShape() const;

private:
  int64_t axis;
  TensorInfo gradOutInfo;
  std::vector<size_t> gradOutShape;
};

} // namespace popx
} // namespace popart

#endif
