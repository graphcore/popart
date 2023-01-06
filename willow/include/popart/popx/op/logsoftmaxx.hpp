// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_LOGSOFTMAXX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_LOGSOFTMAXX_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <poplar/Tensor.hpp>
#include <popart/popx/op/elementwisex.hpp>

#include "popart/popx/debugcontextx.hpp"

namespace poplar {
class Graph;

namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

class LogSoftmaxComputex : public EwuComputex {

public:
  LogSoftmaxComputex(int64_t ax, const std::vector<size_t> &os)
      : axis(ax), outShape(os) {}

  poplar::Tensor outplace(poplar::program::Sequence &,
                          poplar::Graph &,
                          const poplar::Tensor &,
                          const poplar::DebugNameAndId &,
                          const std::string &) const final;

  void inplace(poplar::program::Sequence &,
               poplar::Graph &,
               const poplar::Tensor &,
               const poplar::DebugNameAndId &,
               const std::string &) const final;

  poplar::Tensor reshape(const poplar::Tensor &) const final;

private:
  int64_t axis;
  std::vector<size_t> outShape;
};

class LogSoftmaxOpx : public ElementWiseUnaryOutplaceOpx {
public:
  LogSoftmaxOpx(Op *, Devicex *);
};

class LogSoftmaxInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  LogSoftmaxInplaceOpx(Op *, Devicex *);
};

class LogSoftmaxGradOpx : public ElementWiseUnaryOpx {
public:
  LogSoftmaxGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

  poplar::Tensor cloneNcopyGrouped(poplar::program::Sequence &s,
                                   const poplar::Tensor &t) const;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_LOGSOFTMAXX_HPP_
