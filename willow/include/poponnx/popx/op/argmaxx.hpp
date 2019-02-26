#ifndef GUARD_NEURALNET_ARGMINX_HPP
#define GUARD_NEURALNET_ARGMINX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/argextremax.hpp>

namespace poponnx {

namespace popx {

class ArgMaxOpx : public ArgExtremaOpx {
public:
  using ArgExtremaOpx::ArgExtremaOpx;

private:
  poplar::Tensor selectSlice(const poplar::Tensor &sorted,
                             unsigned axis) const final;
};

} // namespace popx
} // namespace poponnx

#endif
