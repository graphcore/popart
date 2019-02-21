
#include <poponnx/error.hpp>
#include <poponnx/op/max.hpp>
#include <poponnx/popx/op/maxx.hpp>
#include <poponnx/popx/opxmanager.hpp>
#include <poponnx/tensorindex.hpp>

#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/Reduce.hpp>

namespace pe = popops::expr;

namespace poponnx {
namespace popx {

MaxOpx::MaxOpx(Op *op, Devicex *devicex) : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<MaxOp>(op, {Onnx::Operators::Max_8, Onnx::Operators::Max_6});
}

void MaxOpx::grow(poplar::program::Sequence &prog) const {

  auto outTensor = cloneNcopy(prog, get(inId(0)));

  if (op_p->input->n() > 1) {

    for (int i = 1; i < op_p->input->n(); ++i) {
      outTensor = popops::map(graph(),
                              popops::expr::BinaryOpType::MAXIMUM,
                              outTensor,
                              get(inId(i)),
                              prog,
                              idStr());
    }
  }

  insert(outId(MaxOp::getOutIndex()), outTensor);
}

MaxGradOpx::MaxGradOpx(Op *op_, Devicex *devicex_) : Opx(op_, devicex_) {}

void MaxGradOpx::grow(poplar::program::Sequence &prog) const {
  // Create a mask of the max input tensor. Set a element to 1 if it is
  // the maximum element value of all inputs (i.e. is in the fwd output) else 0
  // 1. Subtract the input of the forward op tensor from the out of the
  // forward op.
  //    We will be left with '0' of elements are the maximum in the input tensor
  //    and all other values < 0
  // 2. Signum the result to give a tensor of 0's and -1's.
  // 3. Add 1 from the result to give a mask tensor
  auto mask =
      popops::map(graph(),
                  pe::Add(pe::Signum(pe::Sub(pe::_1, pe::_2)), pe::Const(1)),
                  {get(inId(MaxGradOp::getFwdInIndex())),
                   get(inId(MaxGradOp::getFwdOutInIndex()))},
                  prog,
                  idStr());

  // Multiple the mask by the grad
  auto result = popops::map(graph(),
                            pe::Mul(pe::_1, pe::_2),
                            {mask, get(inId(MaxGradOp::getGradInIndex()))},
                            prog,
                            idStr());

  auto shapeOfOutputOfFwdOp = inInfo(MaxGradOp::getFwdOutInIndex()).shape();
  auto shapeOfInputToFwdOp  = inInfo(MaxGradOp::getFwdInIndex()).shape();

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
  insert(outId(MaxGradOp::getOutIndex()),
         out.reshape(outInfo(MaxGradOp::getOutIndex()).shape_szt()));
}

namespace {
OpxCreator<MaxOpx> maxOpxCreator({Onnx::Operators::Max_6,
                                  Onnx::Operators::Max_8});
OpxCreator<MaxGradOpx> maxGradOpxCreator(Onnx::GradOperators::MaxGrad);
} // namespace

} // namespace popx
} // namespace poponnx
