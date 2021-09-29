// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LOGSOFTMAXX_HPP
#define GUARD_NEURALNET_LOGSOFTMAXX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {

namespace popx {

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
};

} // namespace popx
} // namespace popart

#endif
