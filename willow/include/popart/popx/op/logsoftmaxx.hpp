// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_LOGSOFTMAXX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_LOGSOFTMAXX_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <snap/Tensor.hpp>
#include <string>
#include <vector>
#include <popart/popx/op/elementwisex.hpp>

#include "popart/popx/debugcontextx.hpp"

namespace snap {
class Graph;

namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

class LogSoftmaxComputex : public EwuComputex {

public:
  LogSoftmaxComputex(int64_t ax, const std::vector<size_t> &os)
      : axis(ax), outShape(os) {}

  snap::Tensor outplace(snap::program::Sequence &,
                        snap::Graph &,
                        const snap::Tensor &,
                        const poplar::DebugNameAndId &,
                        const std::string &) const final;

  void inplace(snap::program::Sequence &,
               snap::Graph &,
               const snap::Tensor &,
               const poplar::DebugNameAndId &,
               const std::string &) const final;

  snap::Tensor reshape(const snap::Tensor &) const final;

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
  void grow(snap::program::Sequence &) const final;

  snap::Tensor cloneNcopyGrouped(snap::program::Sequence &s,
                                 const snap::Tensor &t) const;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_LOGSOFTMAXX_HPP_
