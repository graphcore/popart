#include <popops/ElementWise.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/min.hpp>
#include <poponnx/popx/op/minx.hpp>
#include <poponnx/popx/opxmanager.hpp>
#include <poponnx/tensorindex.hpp>
#include <poponnx/util.hpp>

#include <popops/Expr.hpp>
#include <popops/Reduce.hpp>

namespace pe = popops::expr;

namespace poponnx {
namespace popx {

MinOpx::MinOpx(Op *op, Devicex *devicex) : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<MinOp>(op, {Onnx::Operators::Min_8, Onnx::Operators::Min_6});
}

void MinOpx::grow(poplar::program::Sequence &prog) const {
  auto outTensor = cloneNcopy(prog, get(inId(0)));

  if (op_p->input->n() > 1) {

    for (int i = 1; i < op_p->input->n(); ++i) {
      outTensor = popops::map(graph(),
                              popops::expr::BinaryOpType::MINIMUM,
                              outTensor,
                              get(inId(i)),
                              prog,
                              idStr());
    }
  }

  insert(outId(MinOp::getOutIndex()), outTensor);
}

MinArgGradOpx::MinArgGradOpx(Op *op_, Devicex *devicex_) : Opx(op_, devicex_) {}

void MinArgGradOpx::grow(poplar::program::Sequence &prog) const {

  // Create a mask of the min input tensor. Set a element to 1 if it is
  // the minimum element value of all inputs (i.e. is in the fwd output) else 0
  // 1. Subtract the output of the forward op tensor from the input of the
  // forward op.
  //    We will be left with '0' of elements are the minimum in the input tensor
  //    and all other values < 0
  // 2. Signum the result to give a tensor of 0's and -1's.
  // 3. Add 1 to the result to give a mask tensor
  auto mask =
      popops::map(graph(),
                  pe::Add(pe::Signum(pe::Sub(pe::_1, pe::_2)), pe::Const(1)),
                  {get(inId(MinArgGradOp::getFwdOutInIndex())),
                   get(inId(MinArgGradOp::getFwdInIndex()))},
                  prog,
                  idStr());

  // Multiple the mask by the grad input
  auto result = popops::map(graph(),
                            pe::Mul(pe::_1, pe::_2),
                            {mask, get(inId(MinArgGradOp::getGradInIndex()))},
                            prog,
                            idStr());

  auto shapeOfOutputOfFwdOp = inInfo(MinArgGradOp::getFwdOutInIndex()).shape();
  auto shapeOfInputToFwdOp  = inInfo(MinArgGradOp::getFwdInIndex()).shape();

  // Create the axes to reduce along.
  std::vector<int64_t> axes =
      npReductionAxis(shapeOfInputToFwdOp, shapeOfOutputOfFwdOp);

  // Remove axes from the result that were not present ( or 1) in the input to
  // the fwd op
  auto out = popops::reduce(graph(),
                            result,
                            vXtoY<int64_t, std::size_t>(axes),
                            {popops::Operation::ADD},
                            prog,
                            idStr());

  // Reshape the output, to add 1's if needed
  insert(outId(MinArgGradOp::getOutIndex()),
         out.reshape(outInfo(MinArgGradOp::getOutIndex()).shape_szt()));
}

namespace {
OpxCreator<MinOpx> minOpxCreator({Onnx::Operators::Min_6,
                                  Onnx::Operators::Min_8});
OpxCreator<MinArgGradOpx> minGradOpxCreator(Onnx::GradOperators::MinArgGrad);
} // namespace

} // namespace popx
} // namespace poponnx
