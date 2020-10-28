// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LOG1PX_HPP
#define GUARD_NEURALNET_LOG1PX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {

namespace popx {

class Log1pComputex : public EwuComputex {

public:
  Log1pComputex() = default;

  poplar::Tensor outplace(poplar::program::Sequence &,
                          poplar::Graph &,
                          const poplar::Tensor &,
                          const std::string &) const final;

  void inplace(poplar::program::Sequence &,
               poplar::Graph &,
               const poplar::Tensor &,
               const std::string &) const final;

  static std::unique_ptr<EwuComputex> get() {
    return std::unique_ptr<EwuComputex>(new Log1pComputex());
  }
};

class Log1pOpx : public ElementWiseUnaryOutplaceOpx {
public:
  Log1pOpx(Op *, Devicex *);
};

class Log1pInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  Log1pInplaceOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif
