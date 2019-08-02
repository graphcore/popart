#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/Reduce.hpp>

#include <memory>
#include <popart/error.hpp>
#include <popart/op/sum.hpp>
#include <popart/popx/op/sumx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensorindex.hpp>

#include <queue>

namespace pe = popops::expr;

namespace popart {
namespace popx {

SumOpx::SumOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<SumOp>(op, {Onnx::Operators::Sum_6, Onnx::Operators::Sum_8});
}

void SumOpx::grow(poplar::program::Sequence &prog) const {

  SumOp &sumOp = getOp<SumOp>();

  // The input tensors
  std::vector<poplar::Tensor> inputs;

  // The "owner" of all expr nodes
  std::vector<std::unique_ptr<popops::expr::Expr>> exprs;

  // The queue of expr nodes to be reduced
  std::queue<popops::expr::Expr *> expr;

  // Add the input tensors as placeholders to the expression
  for (int i = 0; i < sumOp.input->n(); ++i) {
    inputs.push_back(getInTensor(i));
    exprs.push_back(std::make_unique<popops::expr::PlaceHolder>(i + 1));
    expr.push(exprs.back().get());
  }

  // Build a fairly balanced binary tree
  while (expr.size() > 1) {
    auto &a = *expr.front();
    expr.pop();
    auto &b = *expr.front();
    expr.pop();

    exprs.push_back(std::make_unique<popops::expr::Add>(a, b));
    expr.push(exprs.back().get());
  }

  // Compute the sum
  auto sum = popops::map(graph(), *expr.front(), inputs, prog);
  setOutTensor(SumOp::getOutIndex(), sum);
}

InputCreatorType SumOpx::getInputCreatorType(InIndex index) const {
  // CANUNWIND if doing a series of adds.
  // Check shape doesn't change due to numpy-style broadcasting.
  // Design choice: even without broadcasting, it is possible for the
  // two inputs (of same shape) have different layout.
  // The poplar binary op can choose the layout of the output to take
  // the layout of either input.
  // However, let's layout both inputs in the same way. That way we can
  // definitely unwind through this opx, and it will also be efficient
  // when performing the op.
  if (op_p->inInfo(index) == op_p->outInfo(SumOp::getOutIndex())) {
    return InputCreatorType::CANUNWIND;
  } else {
    return InputCreatorType::DEADEND;
  }
}

poplar::Tensor
SumOpx::unwindTensorLayout(poplar::Tensor tensor, InIndex, OutIndex) const {
  return tensor;
}

SumArgGradOpx::SumArgGradOpx(Op *op_, Devicex *devicex_) : Opx(op_, devicex_) {}

void SumArgGradOpx::grow(poplar::program::Sequence &prog) const {
  auto gradOp = getOp<SumArgGradOp>();

  auto shapeOfInputToBwdOp = inInfo(VariadicGradOp::getGradInIndex()).shape();
  auto shapeOfInputToFwdOp = gradOp.getFwdInputInfo().shape();

  // Create the axes to reduce along.
  std::vector<int64_t> axes =
      npReductionAxis(shapeOfInputToFwdOp, shapeOfInputToBwdOp);

  // Remove axes from the result that were not present ( or 1) in the input to
  // the fwd op
  auto out = popops::reduce(graph(),
                            getInTensor(SumArgGradOp::getGradInIndex()),
                            vXtoY<int64_t, std::size_t>(axes),
                            {popops::Operation::ADD},
                            prog,
                            idStr());

  logging::info("{} Shape of SumArgGradOpx output {} {}",
                out,
                out.shape(),
                outInfo(SumArgGradOp::getOutIndex()).shape_szt());

  // Reshape the output, to add 1's if needed
  setOutTensor(SumArgGradOp::getOutIndex(),
               out.reshape(outInfo(SumArgGradOp::getOutIndex()).shape_szt()));
}

namespace {
OpxCreator<SumOpx> sumOpxCreator({Onnx::Operators::Sum_6,
                                  Onnx::Operators::Sum_8});

OpxCreator<SumArgGradOpx> sumGradOpxCreator(Onnx::GradOperators::SumArgGrad);
} // namespace

} // namespace popx
} // namespace popart
