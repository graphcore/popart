// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_ARGMAXX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_ARGMAXX_HPP_

#include <snap/Tensor.hpp>
#include <popart/popx/op/argextremax.hpp>

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {

namespace popx {

class ArgMaxOpx : public ArgExtremaOpx {
public:
  using ArgExtremaOpx::ArgExtremaOpx;

private:
  snap::Tensor extremaOp(snap::program::Sequence &,
                         const snap::Tensor &) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_ARGMAXX_HPP_
