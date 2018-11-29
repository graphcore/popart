#include <poponnx/error.hpp>
#include <poponnx/op/matmul.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/op/matmulx.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorinfo.hpp>
#include <poponnx/util.hpp>

#include <poplin/MatMul.hpp>
#include <popops/Reduce.hpp>

namespace poponnx {
namespace popx {

MatMulOpx::MatMulOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::MATMUL) {
    throw error("cannot create MatMulOpx from " + op->op_type());
  }
}

std::vector<std::size_t> MatMulOpx::onnxShapeToPoplar(const Shape &shape) {
  std::size_t m      = shape[shape.size() - 2];
  std::size_t n      = shape[shape.size() - 1];
  std::size_t stacks = std::accumulate(
      shape.begin(), shape.end() - 2, 1, std::multiplies<int64_t>());

  return {stacks, m, n};
}

std::vector<std::size_t> MatMulOpx::getLhsInputShape() const {
  auto matmul = getMatMulOp();

  return MatMulOpx::onnxShapeToPoplar(matmul->lhsBroadcastShape());
}

std::vector<std::size_t> MatMulOpx::getLhsInputAllocShape() const {
  auto matmul = getMatMulOp();

  auto shape = matmul->lhsIn()->info.shape();

  if (shape.size() == 1) {
    shape.insert(shape.begin(), 1);
  }

  if (shape.size() == 2) {
    shape.insert(shape.begin(), 1);
  }

  return MatMulOpx::onnxShapeToPoplar(shape);
}

std::vector<std::size_t> MatMulOpx::getLhsInputCoallocShape() const {
  auto lhs_shape = getLhsInputAllocShape();
  auto rhs_shape = getRhsInputAllocShape();

  rhs_shape.front() = lhs_shape.front();

  return rhs_shape;
}

std::vector<std::size_t> MatMulOpx::getRhsInputShape() const {
  auto matmul = getMatMulOp();

  return MatMulOpx::onnxShapeToPoplar(matmul->rhsBroadcastShape());
}

std::vector<std::size_t> MatMulOpx::getRhsInputAllocShape() const {
  auto matmul = getMatMulOp();

  auto shape = matmul->rhsIn()->info.shape();

  if (shape.size() == 1) {
    shape.push_back(1);
  }

  if (shape.size() == 2) {
    shape.insert(shape.begin(), 1);
  }

  return MatMulOpx::onnxShapeToPoplar(shape);
}

std::vector<std::size_t> MatMulOpx::getRhsInputCoallocShape() const {
  auto lhs_shape = getLhsInputAllocShape();
  auto rhs_shape = getRhsInputAllocShape();

  lhs_shape.front() = rhs_shape.front();

  return lhs_shape;
}

std::vector<std::size_t> MatMulOpx::getOutputShape() const {
  auto matmul = getMatMulOp();

  return MatMulOpx::onnxShapeToPoplar(matmul->output.tensor(0)->info.shape());
}

void MatMulOpx::grow(poplar::program::Sequence &prog) const {
  auto matmul = getMatMulOp();

  auto a = broadcast(matmul->lhsBroadcastShape(), getMatMulOp()->lhsIn()->id);
  auto b = broadcast(matmul->rhsBroadcastShape(), getMatMulOp()->rhsIn()->id);

  a = a.reshape(getLhsInputShape());
  b = b.reshape(getRhsInputShape());

  auto outTensor = poplin::matMulGrouped(graph(),            // graph
                                         a,                  // A
                                         b,                  // B
                                         prog,               // prog
                                         idStr(),            // debugPrefix
                                         dv_p->fwdMmOptions, // options
                                         &dv_p->matmulCache  // cache
  );

  insert(outId(0),
         outTensor.reshape(matmul->output.tensor(0)->info.shape_szt()));
}

MatMulOp *MatMulOpx::getMatMulOp() const {
  return dynamic_cast<MatMulOp *>(op_p);
}

bool MatMulOpx::canCreateInput(int) const { return true; }

poplar::Tensor MatMulOpx::createInput(int index) const {
  auto matmul = getMatMulOp();

  if (index == MatMulOp::getLhsInputIndex()) {
    return poplin::createMatMulGroupedInputLHS(
               graph(),
               popType(getMatMulOp()->lhsIn()->info.dataType()),
               getLhsInputAllocShape(),
               getLhsInputCoallocShape(),
               idStr(),
               dv_p->fwdMmOptions,
               &dv_p->matmulCache)
        .reshape(matmul->lhsIn()->info.shape_szt());
  } else if (index == MatMulOp::getRhsInputIndex()) {
    return poplin::createMatMulGroupedInputRHS(
               graph(),
               popType(getMatMulOp()->rhsIn()->info.dataType()),
               getRhsInputCoallocShape(),
               getRhsInputAllocShape(),
               idStr(),
               dv_p->fwdMmOptions,
               &dv_p->matmulCache)
        .reshape(matmul->rhsIn()->info.shape_szt());
  } else {
    throw error("matmul opx cannot create tensor for index " +
                std::to_string(index));
  }
}

bool MatMulOpx::createsEquiv(int ind0, Opx *opx1, int ind1) const {

  if (opx1->op_p->opType != OpType::MATMUL)
    return false;

  if (ind0 != ind1)
    return false;

  // Test that the shapes and types of inputs and outpus of the two ops are the
  // same
  // TODO : Could optimzie this to not use the either lhs or rhs
  MatMulOpx *rhs = dynamic_cast<MatMulOpx *>(opx1);
  if (getMatMulOp()->lhsIn()->info != rhs->getMatMulOp()->lhsIn()->info ||
      getMatMulOp()->rhsIn()->info != rhs->getMatMulOp()->rhsIn()->info ||
      getMatMulOp()->out()->info != rhs->getMatMulOp()->out()->info) {
    return false;
  }

  return true;
}

std::vector<TensorId> MatMulOpx::mustExistBeforeCreate(int) const { return {}; }

MatMulLhsGradOpx::MatMulLhsGradOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  if (op_p->opType != OpType::MATMULLHSGRAD) {
    throw error("cannot create MatMulLhsGradOpx from " + op_p->op_type());
  }
}

void MatMulLhsGradOpx::grow(poplar::program::Sequence &prog) const {
  auto a = broadcast(getGradInputBroadcastShape(),
                     inId(getMatMulLhsGradOp()->getGradInputIndex()));
  auto b = broadcast(getRhsInputBroadcastShape(),
                     inId(getMatMulLhsGradOp()->getRhsInputIndex()));

  a = a.reshape(getGradInputShape());
  b = b.reshape(getRhsInputShape())
          .dimShuffle({0, 2, 1}); // Transpose matrix stack

  auto outTensor = poplin::matMulGrouped(graph(),               // graph
                                         a,                     // A
                                         b,                     // B
                                         prog,                  // prog
                                         idStr(),               // debugPrefix
                                         dv_p->bwdMmLhsOptions, // options
                                         &dv_p->matmulCache     // cache
  );

  // If the number of element are the same, then we only need a reshape.
  if (outTensor.numElements() != outInfo(0).nelms()) {
    outTensor = popops::reduce(graph(),
                               outTensor.reshape(getOutputShape()),
                               getOutputReductionAxes(),
                               {popops::Operation::ADD},
                               prog);
  }

  insert(outId(0), outTensor.reshape(outInfo(0).shape_szt()));
}

MatMulLhsGradOp *MatMulLhsGradOpx::getMatMulLhsGradOp() const {
  return dynamic_cast<MatMulLhsGradOp *>(op_p);
}

std::vector<std::size_t> MatMulLhsGradOpx::getGradInputShape() const {
  return MatMulOpx::onnxShapeToPoplar(getGradInputBroadcastShape());
}

Shape MatMulLhsGradOpx::getGradInputBroadcastShape() const {
  auto matmul_lgrad = getMatMulLhsGradOp();
  return matmul_lgrad->getGradInputShape();
}

std::vector<std::size_t> MatMulLhsGradOpx::getRhsInputShape() const {
  return MatMulOpx::onnxShapeToPoplar(getRhsInputBroadcastShape());
}

Shape MatMulLhsGradOpx::getRhsInputBroadcastShape() const {
  auto matmul_lgrad = getMatMulLhsGradOp();

  const auto rhs_shape = matmul_lgrad->getRhsInputShape();
  auto grad_shape      = matmul_lgrad->getGradInputShape();

  std::copy(rhs_shape.end() - 2, rhs_shape.end(), grad_shape.end() - 2);

  return grad_shape;
}

std::vector<std::size_t> MatMulLhsGradOpx::getOutputShape() const {
  auto matmul_lgrad = getMatMulLhsGradOp();

  const auto rhs_shape = matmul_lgrad->getRhsInputShape();
  auto grad_shape      = matmul_lgrad->getGradInputShape();

  grad_shape.back() = rhs_shape[rhs_shape.size() - 2];

  return vXtoY<int64_t, std::size_t>(grad_shape);
}

std::vector<std::size_t> MatMulLhsGradOpx::getOutputReductionAxes() const {
  auto matmul_lgrad = getMatMulLhsGradOp();

  return vXtoY<int64_t, std::size_t>(
      npReductionAxis(matmul_lgrad->getOutputShape(),
                      vXtoY<std::size_t, int64_t>(getOutputShape())));
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

void MatMulRhsGradOpx::grow(poplar::program::Sequence &prog) const {
  auto a = broadcast(getLhsInputBroadcastShape(),
                     inId(getMatMulRhsGradOp()->getLhsInputIndex()));
  auto b = broadcast(getGradInputBroadcastShape(),
                     inId(getMatMulRhsGradOp()->getGradInputIndex()));

  a = a.reshape(getLhsInputShape())
          .dimShuffle({0, 2, 1}); // Transpose matrix stack
  b = b.reshape(getGradInputShape());

  auto outTensor = poplin::matMulGrouped(graph(),               // graph
                                         a,                     // A
                                         b,                     // B
                                         prog,                  // prog
                                         idStr(),               // debugPrefix
                                         dv_p->bwdMmRhsOptions, // options
                                         &dv_p->matmulCache     // cache
  );

  // If the number of element are the same, then we only need a reshape.
  if (outTensor.numElements() != outInfo(0).nelms()) {
    outTensor = popops::reduce(graph(),
                               outTensor.reshape(getOutputShape()),
                               getOutputReductionAxes(),
                               {popops::Operation::ADD},
                               prog);
  }

  insert(outId(0), outTensor.reshape(outInfo(0).shape_szt()));
}

std::vector<std::size_t> MatMulRhsGradOpx::getLhsInputShape() const {
  return MatMulOpx::onnxShapeToPoplar(getLhsInputBroadcastShape());
}

Shape MatMulRhsGradOpx::getLhsInputBroadcastShape() const {
  auto matmul_rgrad = getMatMulRhsGradOp();

  const auto lhs_shape = matmul_rgrad->getLhsInputShape();
  auto grad_shape      = matmul_rgrad->getGradInputShape();

  std::copy(lhs_shape.end() - 2, lhs_shape.end(), grad_shape.end() - 2);

  return grad_shape;
}

std::vector<std::size_t> MatMulRhsGradOpx::getGradInputShape() const {
  return MatMulOpx::onnxShapeToPoplar(getGradInputBroadcastShape());
}

Shape MatMulRhsGradOpx::getGradInputBroadcastShape() const {
  auto matmul_rgrad = getMatMulRhsGradOp();
  return matmul_rgrad->getGradInputShape();
}

std::vector<std::size_t> MatMulRhsGradOpx::getOutputShape() const {
  auto matmul_rgrad = getMatMulRhsGradOp();

  const auto lhs_shape = matmul_rgrad->getLhsInputShape();
  auto grad_shape      = matmul_rgrad->getGradInputShape();

  grad_shape[grad_shape.size() - 2] = lhs_shape.back();

  return vXtoY<int64_t, std::size_t>(grad_shape);
}

std::vector<std::size_t> MatMulRhsGradOpx::getOutputReductionAxes() const {
  auto matmul_rgrad = getMatMulRhsGradOp();

  return vXtoY<int64_t, std::size_t>(
      npReductionAxis(matmul_rgrad->getOutputShape(),
                      vXtoY<std::size_t, int64_t>(getOutputShape())));
}

} // namespace popx
} // namespace poponnx
