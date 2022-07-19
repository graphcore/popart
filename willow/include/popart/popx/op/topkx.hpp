// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_TOPKX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_TOPKX_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>
#include <popart/popx/op/basesortx.hpp>

#include "popart/popx/popopx.hpp"
#include "popart/tensorinfo.hpp"

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

class TopKOpx : public BaseSortOpx {
public:
  TopKOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

private:
  unsigned K;
};

class TopKGradOpx : public PopOpx {
public:
  TopKGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
  // The info of the output of this Op
  const std::vector<size_t> &getGradOutShape() const;

private:
  int64_t axis;
  TensorInfo gradOutInfo;
  std::vector<size_t> gradOutShape;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_TOPKX_HPP_
