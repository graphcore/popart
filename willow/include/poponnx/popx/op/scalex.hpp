#ifndef GUARD_NEURALNET_SCALEX_HPP
#define GUARD_NEURALNET_SCALEX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/elementwisex.hpp>

namespace poponnx {

namespace popx {

class ScaleComputex : public EwuComputex {

public:
  ScaleComputex(double x) : scale_factor(x) {}

  poplar::Tensor outplace(poplar::program::Sequence &,
                          poplar::Graph &,
                          const poplar::Tensor &tensor,
                          const std::string &) const final;

  void inplace(poplar::program::Sequence &,
               poplar::Graph &,
               const poplar::Tensor &,
               const std::string &) const final;

  static std::unique_ptr<EwuComputex> get(float x) {
    return std::unique_ptr<EwuComputex>(
        new ScaleComputex(static_cast<double>(x)));
  }

  poplar::Tensor getScaleTensor(const poplar::Type &type,
                                poplar::Graph &graph) const;

  static float getFromScaleOp(Op *op);
  static float getFromScaleInplaceOp(Op *op);

private:
  double scale_factor;
};

class ScaleOpx : public ElementWiseUnaryOutplaceOpx {
public:
  ScaleOpx(Op *, Devicex *);
};

class ScaleInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  ScaleInplaceOpx(Op *, Devicex *);
};

class ScaleGradOpx : public ScaleOpx {
public:
  ScaleGradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace poponnx

#endif
