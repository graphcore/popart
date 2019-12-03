#ifndef GUARD_NEURALNET_MATMULX_HPP
#define GUARD_NEURALNET_MATMULX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {

class MatMulOp;
class MatMulLhsGradOp;
class MatMulRhsGradOp;

namespace popx {

class MatMulOpx : public Opx {
public:
  MatMulOpx(Op *, Devicex *);
  ~MatMulOpx() override = default;

  poplar::Tensor createInput(InIndex index,
                             const std::string &name) const final;
  InputCreatorType getInputCreatorType(InIndex index) const final;
  bool createsEquiv(int, const Opx *, int) const final;
  std::vector<TensorId> mustExistBeforeCreate(InIndex index0) const final;

  MatMulOp *getMatMulOp() const;
  void grow(poplar::program::Sequence &) const final;

  static std::vector<std::size_t> onnxShapeToPoplar(const Shape &shape);

private:
  // The ONNX tensor shape
  std::vector<std::size_t> getOutputShape() const;
};

} // namespace popx
} // namespace popart

#endif
