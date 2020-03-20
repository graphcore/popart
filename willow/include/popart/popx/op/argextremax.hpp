// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ARGEXTREMAX_HPP
#define GUARD_NEURALNET_ARGEXTREMAX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {

namespace popx {

class ArgExtremaOpx : public Opx {
public:
  ArgExtremaOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;

private:
  virtual poplar::Tensor extremaOp(poplar::program::Sequence &,
                                   const poplar::Tensor &) const = 0;
};

} // namespace popx
} // namespace popart

#endif
