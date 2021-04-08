// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SHAPEDDROPOUTX_HPP
#define GUARD_NEURALNET_SHAPEDDROPOUTX_HPP

#include <popart/popx/opx.hpp>

namespace popart {
namespace popx {

class ShapedDropoutOpx : public Opx {
public:
  ShapedDropoutOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;

private:
  poplar::Tensor getReferenceTensor() const;
};

} // namespace popx
} // namespace popart

#endif
