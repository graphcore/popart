#ifndef GUARD_NEURALNET_FLOORX_HPP
#define GUARD_NEURALNET_FLOORX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/elementwisex.hpp>

namespace poponnx {

namespace popx {

class FloorComputex : public EwuComputex {

public:
  FloorComputex() {}

  poplar::Tensor outplace(poplar::program::Sequence &,
                          poplar::Graph &,
                          const poplar::Tensor &tensor,
                          const std::string &) const final;

  void inplace(poplar::program::Sequence &,
               poplar::Graph &,
               const poplar::Tensor &,
               const std::string &) const final;

  static std::unique_ptr<EwuComputex> get() {
    return std::unique_ptr<EwuComputex>(new FloorComputex());
  }
};

class FloorOpx : public ElementWiseUnaryOutplaceOpx {
public:
  FloorOpx(Op *, Devicex *);
};

class FloorInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  FloorInplaceOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace poponnx

#endif
