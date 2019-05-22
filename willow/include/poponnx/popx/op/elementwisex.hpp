#ifndef GUARD_NEURALNET_ELEMENTWISEUNARYX_HPP
#define GUARD_NEURALNET_ELEMENTWISEUNARYX_HPP

#include <poponnx/popx/opx.hpp>

namespace poponnx {
namespace popx {

// A base class with functions for computing in-place and
// out-of place element-wise unary operations
class EwuComputex {
public:
  EwuComputex()          = default;
  virtual ~EwuComputex() = default;

  virtual poplar::Tensor outplace(poplar::program::Sequence &,
                                  poplar::Graph &,
                                  const poplar::Tensor &,
                                  const std::string &) const = 0;

  virtual void inplace(poplar::program::Sequence &,
                       poplar::Graph &,
                       const poplar::Tensor &t,
                       const std::string &) const = 0;

  poplar::Tensor cloneNcopy(poplar::program::Sequence &,
                            poplar::Graph &,
                            const poplar::Tensor &) const;

  // certain elementwise unary ops may reshape the input tensor (eg Softmax)
  virtual poplar::Tensor reshape(const poplar::Tensor &t) const { return t; }
};

// Base class for elementwise unary operations
class ElementWiseUnaryOpx : public Opx {
public:
  ElementWiseUnaryOpx(Op *, Devicex *);
  InputCreatorType getInputCreatorType(InIndex) const override;
  poplar::Tensor
      unwindTensorLayout(poplar::Tensor, InIndex, OutIndex) const override;
};

// non-inplace
class ElementWiseUnaryOutplaceOpx : public ElementWiseUnaryOpx {
public:
  ElementWiseUnaryOutplaceOpx(Op *,
                              Devicex *,
                              std::unique_ptr<EwuComputex> cx_);
  void grow(poplar::program::Sequence &) const final;

private:
  std::unique_ptr<EwuComputex> cx;
};

// inplace
class ElementWiseUnaryInplaceOpx : public ElementWiseUnaryOpx {
public:
  ElementWiseUnaryInplaceOpx(Op *op,
                             Devicex *devx,
                             std::unique_ptr<EwuComputex> cx_)
      : ElementWiseUnaryOpx(op, devx), cx(std::move(cx_)) {}
  void grow(poplar::program::Sequence &prog) const final;

private:
  std::unique_ptr<EwuComputex> cx;
};

// Base class for elementwise binary operations
class ElementWiseBinaryOpx : public Opx {
public:
  ElementWiseBinaryOpx(Op *, Devicex *);
  InputCreatorType getInputCreatorType(InIndex) const override;
  poplar::Tensor
  unwindTensorLayout(poplar::Tensor tensor, InIndex, OutIndex) const override;
};

} // namespace popx
} // namespace poponnx

#endif
