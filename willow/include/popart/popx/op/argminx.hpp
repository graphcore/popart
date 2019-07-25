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
  poplar::Tensor extremaOp(poplar::program::Sequence &,
                           const poplar::Tensor &) const final;
};

} // namespace popx
} // namespace popart

#endif
