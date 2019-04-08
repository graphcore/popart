#ifndef GUARD_NEURALNET_EXPX_HPP
#define GUARD_NEURALNET_EXPX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/elementwisex.hpp>

namespace poponnx {

namespace popx {

class ExpComputex : public EwuComputex {

public:
  ExpComputex() = default;

  poplar::Tensor outplace(poplar::program::Sequence &,
                          poplar::Graph &,
                          const poplar::Tensor &) const final;

  void inplace(poplar::program::Sequence &,
               poplar::Graph &,
               const poplar::Tensor &) const final;

  static std::unique_ptr<EwuComputex> get() {
    return std::unique_ptr<EwuComputex>(new ExpComputex);
  }
};

class ExpOpx : public ElementWiseUnaryOutplaceOpx {
public:
  ExpOpx(Op *, Devicex *);
};

class ExpInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  ExpInplaceOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace poponnx

#endif
