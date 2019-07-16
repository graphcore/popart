#include <numeric>
#include <poponnx/error.hpp>
#include <poponnx/op/l1.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/op/l1x.hpp>
#include <poponnx/popx/opxmanager.hpp>
#include <poponnx/tensor.hpp>

#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>

namespace poponnx {
namespace popx {

L1Opx::L1Opx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<L1Op>(op, Onnx::CustomOperators::L1);
}

void L1GradOpx::grow(poplar::program::Sequence &prog) const {
  L1GradOp &l1gradop = getOp<L1GradOp>();

  double lambda = static_cast<double>(l1gradop.l1l()->getLambda());

  // Signum : +1 of positive, -1 if negative, 0 if zero.
  poplar::Tensor signumTensor = popops::map(graph(),
                                            popops::expr::UnaryOpType::SIGNUM,
                                            getInTensor(0),
                                            prog,
                                            debugPrefix("Signum"));

  double scale;
  switch (l1gradop.l1l()->getReductionType()) {
  case ReductionType::SUM: {
    scale = lambda;
    break;
  }
  case ReductionType::MEAN: {
    // Design note: The L1Loss measures the mean absolute error between
    // each element of the input and a (zero) target. The target here is the
    // same size as the input tensor. This is unlike NllLoss, whose target
    // is a 1D tensor of size `batchSize`.
    // As a result the mean reduction is not over the size of the outer
    // dimension, but over the total number of elements in the tensor.
    // This is consistent with pytorch: pytorch.org/docs/stable/nn.html#l1loss
    double totalSamples = static_cast<double>(dv_p->getReplicationFactor()) *
                          static_cast<double>(getInTensor(0).numElements());
    scale = lambda / totalSamples;
    break;
  }
  default: {
    throw error("Unsupported reduction type for Loss {}", idStr());
  }
  }

  auto t_scale =
      getConst(popType(op_p->inInfo(0)), {1}, scale, debugPrefix("scale"));

  // scale the signum tensor by 'scale',
  // so +scale if positive, -scale if negative, 0 if zero
  auto gradTensor = popops::map(graph(),
                                popops::expr::BinaryOpType::MULTIPLY,
                                signumTensor,
                                t_scale,
                                prog,
                                debugPrefix("Multiply"));

  setOutTensor(0, gradTensor);
}

InputCreatorType L1Opx::getInputCreatorType(InIndex) const {
  return InputCreatorType::CANUNWIND;
}

// lambda * sum_{0,..rank-1} |v|
void L1Opx::grow(poplar::program::Sequence &prog) const {
  L1Op &l1op               = getOp<L1Op>();
  poplar::Tensor absTensor = popops::map(graph(),
                                         popops::expr::UnaryOpType::ABSOLUTE,
                                         getInTensor(0),
                                         prog,
                                         debugPrefix("abs"));

  if (absTensor.rank() == 0) {
    throw error("invalid tensor (rank-0) in L1Opx");
  }

  std::vector<size_t> dims(absTensor.rank() - 1);

  // we will reduce over {1,....rank -1}. NOT
  // over dimension 0, which is batch id
  std::iota(dims.begin(), dims.end(), 1);

  double lambda = static_cast<double>(l1op.l1l()->getLambda());

  double scale;
  switch (l1op.l1l()->getReductionType()) {
  case ReductionType::SUM: {
    scale = lambda;
    break;
  }
  case ReductionType::MEAN: {
    double totalSamples = static_cast<double>(dv_p->getReplicationFactor()) *
                          static_cast<double>(getInTensor(0).dim(0));
    scale = lambda / totalSamples;
    break;
  }
  default: {
    throw error("Unsupported reduction type for Loss {}", idStr());
  }
  }

  // t_scale is always expected to be FLOAT, regardless of the input type
  // to the reduction
  auto t_scale = getConst(poplar::FLOAT, {}, scale, debugPrefix("scale"));

  auto reduction = popops::reduce(graph(),
                                  absTensor,
                                  dims,
                                  {popops::Operation::ADD, false, t_scale},
                                  prog,
                                  debugPrefix("add"));

  setOutTensor(0, reduction);
}

L1GradOpx::L1GradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<L1GradOp>(op, Onnx::CustomGradOperators::L1Grad);
}

namespace {
OpxCreator<L1Opx> l1OpxCreator(Onnx::CustomOperators::L1);
OpxCreator<L1GradOpx> l1GradOpxCreator(Onnx::CustomGradOperators::L1Grad);
} // namespace

} // namespace popx
} // namespace poponnx
