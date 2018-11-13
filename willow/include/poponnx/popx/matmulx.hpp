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
  virtual ~MatMulOpx() override = default;

  MatMulOp *getMatMulOp() const;
  virtual void grow() const override final;
};

class MatMulLhsGradOpx : public Opx {
public:
  MatMulLhsGradOpx(Op *, Devicex *);
  virtual ~MatMulLhsGradOpx() override = default;

  MatMulLhsGradOp *getMatMulLhsGradOp() const;
  virtual void grow() const override final;
};

class MatMulRhsGradOpx : public Opx {
public:
  MatMulRhsGradOpx(Op *, Devicex *);
  virtual ~MatMulRhsGradOpx() override = default;

  MatMulRhsGradOp *getMatMulRhsGradOp() const;
  virtual void grow() const override final;
};

} // namespace popx
} // namespace willow

#endif
