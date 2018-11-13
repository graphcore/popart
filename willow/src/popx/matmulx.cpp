#include <poponnx/error.hpp>
#include <poponnx/matmul.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/matmulx.hpp>
#include <poponnx/tensor.hpp>

#include <poplin/MatMul.hpp>

namespace willow {
namespace popx {

MatMulOpx::MatMulOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::MATMUL) {
    throw error("cannot create MatMulOpx from " + op->op_type());
  }
}

void MatMulOpx::grow() const {

  auto outTensor = poplin::matMul(graph(),                         // graph
                                  get(getMatMulOp()->lhsIn()->id), // A
                                  get(getMatMulOp()->rhsIn()->id), // B
                                  dv_p->progs.step(),              // prog
                                  idStr(),            // debugPrefix
                                  dv_p->fwdMmOptions, // options
                                  &dv_p->matmulCache  // cache
  );

  insert(outId(0), outTensor);
}

MatMulOp *MatMulOpx::getMatMulOp() const {
  return dynamic_cast<MatMulOp *>(op_p);
}

MatMulLhsGradOpx::MatMulLhsGradOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  if (op_p->opType != OpType::MATMULLHSGRAD) {
    throw error("cannot create MatMulLhsGradOpx from " + op_p->op_type());
  }
}

void MatMulLhsGradOpx::grow() const {

  auto outTensor = poplin::matMul(
      graph(),                                                         // graph
      get(inId(getMatMulLhsGradOp()->getGradInputIndex())),            // A
      get(inId(getMatMulLhsGradOp()->getRhsInputIndex())).transpose(), // B
      step(),                                                          // prog
      idStr(),               // debugPrefix
      dv_p->bwdMmLhsOptions, // options
      &dv_p->matmulCache     // cache
  );

  insert(outId(0), outTensor);
}

MatMulLhsGradOp *MatMulLhsGradOpx::getMatMulLhsGradOp() const {
  return dynamic_cast<MatMulLhsGradOp *>(op_p);
}

MatMulRhsGradOpx::MatMulRhsGradOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  if (op_p->opType != OpType::MATMULRHSGRAD) {
    throw error("cannot create MatMulRhsGradOpx from " + op_p->op_type());
  }
}

MatMulRhsGradOp *MatMulRhsGradOpx::getMatMulRhsGradOp() const {
  return dynamic_cast<MatMulRhsGradOp *>(op_p);
}

void MatMulRhsGradOpx::grow() const {

  auto outTensor = poplin::matMul(
      graph(),                                                         // graph
      get(inId(getMatMulRhsGradOp()->getLhsInputIndex())).transpose(), // A
      get(inId(getMatMulRhsGradOp()->getGradInputIndex())),            // B
      step(),                                                          // prog
      idStr(),               // debugPrefix
      dv_p->bwdMmRhsOptions, // options
      &dv_p->matmulCache     // cache
  );

  insert(outId(0), outTensor);
}

} // namespace popx
} // namespace willow
