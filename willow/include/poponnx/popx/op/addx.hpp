#ifndef GUARD_NEURALNET_ADDX_HPP
#define GUARD_NEURALNET_ADDX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/reducesumx.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

class AddOp;

namespace popx {

class AddOpx : public Opx {
public:
  AddOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class AddArg0GradOpx : public ReduceSumOpx {
public:
  AddArg0GradOpx(Op *, Devicex *);
};

class AddArg1GradOpx : public ReduceSumOpx {
public:
  AddArg1GradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace poponnx

#endif
