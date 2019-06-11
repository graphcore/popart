#ifndef GUARD_NEURALNET_REDUCEPRODX_HPP
#define GUARD_NEURALNET_REDUCEPRODX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

class ReduceProdOp;

namespace popx {

class ReduceProdOpx : public Opx {
public:
  ReduceProdOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

class ReduceProdGradOpx : public Opx {
public:
  ReduceProdGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

} // namespace popx
} // namespace poponnx

#endif
