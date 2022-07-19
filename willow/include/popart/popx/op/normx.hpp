// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_NORMX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_NORMX_HPP_

#include <algorithm>
#include <cstddef>
#include <snap/Tensor.hpp>
#include <vector>
#include <poplar/Type.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {
class Op;

namespace popx {
class Devicex;
} // namespace popx
} // namespace popart
namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

// Base class for the norm options such as group, instance, batch

namespace poplar {
using Shape = std::vector<std::size_t>;
}

namespace popart {
namespace popx {

class NormOpx : public PopOpx {
public:
  NormOpx(Op *, Devicex *);

protected:
  snap::Tensor convertInvSdToVar(snap::program::Sequence &prog,
                                 const snap::Tensor &invSd,
                                 float epsilon,
                                 const poplar::Type dstType) const;

  snap::Tensor convertVarToInvSd(snap::program::Sequence &prog,
                                 const snap::Tensor &var,
                                 float epsilon,
                                 const poplar::Type dstType) const;

private:
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_NORMX_HPP_
