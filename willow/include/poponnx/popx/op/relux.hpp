#ifndef GUARD_NEURALNET_RELUX_HPP
#define GUARD_NEURALNET_RELUX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/elementwisex.hpp>

namespace poponnx {

class ReluOp;
class ReluInplaceOp;
class ReluGradOp;

namespace popx {

class ReluComputex : public EwuComputex {

public:
  ReluComputex() = default;

  poplar::Tensor outplace(poplar::program::Sequence &,
                          poplar::Graph &,
                          const poplar::Tensor &) const final;

  void inplace(poplar::program::Sequence &,
               poplar::Graph &,
               const poplar::Tensor &) const final;

  static std::unique_ptr<EwuComputex> get() {
    return std::unique_ptr<EwuComputex>(new ReluComputex);
  }
};

class ReluOpx : public ElementWiseUnaryOutplaceOpx {
public:
  ReluOpx(Op *, Devicex *);
};

class ReluInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  ReluInplaceOpx(Op *, Devicex *);
};

class ReluGradOpx : public Opx {
public:
  ReluGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
