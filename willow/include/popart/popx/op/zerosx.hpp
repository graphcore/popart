// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ZEROSX_HPP
#define GUARD_NEURALNET_ZEROSX_HPP

#include <popart/popx/opx.hpp>

namespace popart {

namespace popx {

class ZerosOpx : public Opx {
public:
  ZerosOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
