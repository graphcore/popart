#ifndef GUARD_NEURALNET_INSTANCENORMX_HPP
#define GUARD_NEURALNET_INSTANCENORMX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/normx.hpp>

namespace poponnx {

namespace popx {

class InstanceNormOpx : public NormOpx {
public:
  InstanceNormOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
};

} // namespace popx
} // namespace poponnx

#endif
