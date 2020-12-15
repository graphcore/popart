// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SIGMOIDX_HPP
#define GUARD_NEURALNET_SIGMOIDX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {

namespace popx {

class SigmoidComputex : public EwuComputex {

public:
  SigmoidComputex() = default;

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

  static std::unique_ptr<EwuComputex> get() {
    return std::unique_ptr<EwuComputex>(new SigmoidComputex);
  }
};

class SigmoidOpx : public ElementWiseUnaryOutplaceOpx {
public:
  SigmoidOpx(Op *, Devicex *);
};

class SigmoidInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  SigmoidInplaceOpx(Op *, Devicex *);
};

class SigmoidGradOpx : public ElementWiseUnaryOpx {
public:
  SigmoidGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
