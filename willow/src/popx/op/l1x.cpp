#include <numeric>
#include <poponnx/error.hpp>
#include <poponnx/op/l1.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/op/l1x.hpp>
#include <poponnx/tensor.hpp>

#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>

namespace willow {
namespace popx {

L1Opx::L1Opx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::L1) {
    throw error("cannot create L1Opx from " + op->op_type());
  }
}

void L1GradOpx::grow(poplar::program::Sequence &prog) const {
  L1GradOp *l1gradop = getL1GradOp();
  poplar::Tensor t_lambda =
      dv_p->getConst(popType(op_p->input.tensor(0)->info),
                     {1},
                     static_cast<double>(l1gradop->l1l()->getLambda()));

  // Signum : +1 of positive, -1 if negative, 0 if zero.
  poplar::Tensor signumTensor = popops::map(graph(),
                                            popops::expr::UnaryOpType::SIGNUM,
                                            get(inId(0)),
                                            prog,
                                            "signum/" + inId(0));

  // scale the signum tensor by lambda,
  // so +lambda if positive, -lambda if negative, 0 if zero
  poplar::Tensor gradTensor = popops::map(graph(),
                                          popops::expr::BinaryOpType::MULTIPLY,
                                          signumTensor,
                                          t_lambda,
                                          prog,
                                          "multiply/" + inId(0));

  insert(outId(0), gradTensor);
}

// lambda * sum_{0,..rank-1} |v|
void L1Opx::grow(poplar::program::Sequence &prog) const {
  L1Op *l1op               = getL1Op();
  poplar::Tensor absTensor = popops::map(graph(),
                                         popops::expr::UnaryOpType::ABSOLUTE,
                                         get(inId(0)),
                                         prog,
                                         "abs/" + inId(0));

  if (absTensor.rank() == 0) {
    throw error("invalid tensor (rank-0) in L1Opx");
  }

  std::vector<size_t> dims(absTensor.rank() - 1);

  // we will reduce over {1,....rank -1}. NOT
  // over dimension 0, which is batch id
  std::iota(dims.begin(), dims.end(), 1);

  poplar::Tensor reduction =
      popops::reduce(graph(),
                     absTensor,
                     dims,
                     {popops::Operation::ADD, l1op->l1l()->getLambda()},
                     prog);

  insert(outId(0), reduction);
}

L1Op *L1Opx::getL1Op() const { return dynamic_cast<L1Op *>(op_p); }

L1GradOpx::L1GradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::L1GRAD) {
    throw error("cannot create L1GradOpx from " + op->op_type());
  }
}

L1GradOp *L1GradOpx::getL1GradOp() const {
  return dynamic_cast<L1GradOp *>(op_p);
}

} // namespace popx
} // namespace willow
