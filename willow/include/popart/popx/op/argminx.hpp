// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ARGMINX_HPP
#define GUARD_NEURALNET_ARGMINX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/argextremax.hpp>

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
