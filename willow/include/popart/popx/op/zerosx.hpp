// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ZEROSX_HPP
#define GUARD_NEURALNET_ZEROSX_HPP

#include <popart/popx/popopx.hpp>

namespace popart {

namespace popx {

class ZerosOpx : public PopOpx {
public:
  ZerosOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
