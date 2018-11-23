#ifndef GUARD_NEURALNET_MATMULX_HPP
#define GUARD_NEURALNET_MATMULX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace willow {

class MatMulOp;
class MatMulLhsGradOp;
class MatMulRhsGradOp;

namespace popx {

class MatMulOpx : public Opx {
public:
  MatMulOpx(Op *, Devicex *);
  ~MatMulOpx() override = default;

  poplar::Tensor createInput(int index) const final;
  bool canCreateInput(int index) const final;
  bool createsEquiv(int, Opx *, int) const final;
  std::vector<TensorId> mustExistBeforeCreate(int index0) const final;

  MatMulOp *getMatMulOp() const;
  void grow(poplar::program::Sequence &) const final;
};

class MatMulLhsGradOpx : public Opx {
public:
  MatMulLhsGradOpx(Op *, Devicex *);
  ~MatMulLhsGradOpx() override = default;

  MatMulLhsGradOp *getMatMulLhsGradOp() const;
  void grow(poplar::program::Sequence &) const final;
};

class MatMulRhsGradOpx : public Opx {
public:
  MatMulRhsGradOpx(Op *, Devicex *);
  ~MatMulRhsGradOpx() override = default;

  MatMulRhsGradOp *getMatMulRhsGradOp() const;
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace willow

#endif
