#ifndef GUARD_NEURALNET_ARGMINX_HPP
#define GUARD_NEURALNET_ARGMINX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/argextremax.hpp>

namespace poponnx {

namespace popx {

class ArgMinOpx : public ArgExtremaOpx {
public:
  using ArgExtremaOpx::ArgExtremaOpx;

private:
  poplar::Tensor extremaOp(poplar::program::Sequence &,
                           const poplar::Tensor &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
