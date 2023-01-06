// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_ARGMAXX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_ARGMAXX_HPP_

#include <poplar/Tensor.hpp>
#include <popart/popx/op/argextremax.hpp>

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {

namespace popx {

class ArgMaxOpx : public ArgExtremaOpx {
public:
  using ArgExtremaOpx::ArgExtremaOpx;

private:
  poplar::Tensor extremaOp(poplar::program::Sequence &,
                           const poplar::Tensor &) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_ARGMAXX_HPP_
