// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_NORMX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_NORMX_HPP_

#include <algorithm>
#include <cstddef>
#include <ext/new_allocator.h>
#include <vector>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <popart/popx/opx.hpp>

namespace popart {
class Op;

namespace popx {
class Devicex;
} // namespace popx
} // namespace popart
namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

// Base class for the norm options such as group, instance, batch

namespace poplar {
using Shape = std::vector<std::size_t>;
}

namespace popart {
namespace popx {

class NormOpx : public Opx {
public:
  NormOpx(Op *, Devicex *);

protected:
  poplar::Tensor convertInvSdToVar(poplar::program::Sequence &prog,
                                   const poplar::Tensor &invSd,
                                   float epsilon,
                                   const poplar::Type dstType) const;

  poplar::Tensor convertVarToInvSd(poplar::program::Sequence &prog,
                                   const poplar::Tensor &var,
                                   float epsilon,
                                   const poplar::Type dstType) const;

private:
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_NORMX_HPP_
