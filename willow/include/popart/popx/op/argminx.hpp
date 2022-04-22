// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ARGMINX_HPP
#define GUARD_NEURALNET_ARGMINX_HPP

#include <snap/Tensor.hpp>
#include <popart/popx/op/argextremax.hpp>

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {

namespace popx {

class ArgMinOpx : public ArgExtremaOpx {
public:
  using ArgExtremaOpx::ArgExtremaOpx;

private:
  snap::Tensor extremaOp(snap::program::Sequence &,
                         const snap::Tensor &) const final;
};

} // namespace popx
} // namespace popart

#endif
