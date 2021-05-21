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

  poplar::Tensor outplace(poplar::program::Sequence &,
                          snap::Graph &,
                          const poplar::Tensor &,
                          const poplar::DebugNameAndId &,
                          const std::string &) const final;

  void inplace(poplar::program::Sequence &,
               snap::Graph &,
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
};

} // namespace popx
} // namespace popart

#endif
