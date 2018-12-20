#ifndef GUARD_NEURALNET_AVERAGEPOOLX_HPP
#define GUARD_NEURALNET_AVERAGEPOOLX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

class AveragePoolOp;
class AveragePoolGradOp;

namespace popx {

class AveragePoolOpx : public Opx {
public:
  AveragePoolOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class AveragePoolGradOpx : public Opx {
public:
  AveragePoolGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
