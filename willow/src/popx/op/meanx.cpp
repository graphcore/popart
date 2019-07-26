
#include <popart/error.hpp>
#include <popart/op/mean.hpp>
#include <popart/popx/op/meanx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensorindex.hpp>

#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/Reduce.hpp>

namespace pe = popops::expr;

namespace popart {
namespace popx {

MeanOpx::MeanOpx(Op *op, Devicex *devicex) : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<MeanOp>(op, {Onnx::Operators::Mean_8, Onnx::Operators::Mean_6});
}

void MeanOpx::grow(poplar::program::Sequence &prog) const {
  auto outTensor = cloneNcopy(prog, getInTensor(0));

  if (op_p->input->n() > 1) {

    for (int i = 1; i < op_p->input->n(); ++i) {
      outTensor = popops::map(graph(),
                              popops::expr::BinaryOpType::ADD,
                              outTensor,
                              getInTensor(i),
                              prog,
                              idStr());
    }

    outTensor = popops::map(graph(),
                            pe::Divide(pe::_1, pe::Const(op_p->input->n())),
                            {outTensor},
                            prog,
                            idStr());
  }

  setOutTensor(MeanOp::getOutIndex(), outTensor);
}

MeanArgGradOpx::MeanArgGradOpx(Op *op_, Devicex *devicex_)
    : Opx(op_, devicex_) {}

void MeanArgGradOpx::grow(poplar::program::Sequence &prog) const {
  auto gradOp = getOp<MeanArgGradOp>();

  auto shapeOfInputToBwdOp = inInfo(MeanArgGradOp::getGradInIndex()).shape();
  auto shapeOfInputToFwdOp = gradOp.getFwdInputInfo().shape();

  // Create the axes to reduce along.
  std::vector<int64_t> axes =
      npReductionAxis(shapeOfInputToFwdOp, shapeOfInputToBwdOp);

  // Remove axes from the result that were not present ( or 1) in the input to
  // the fwd op
  auto out = popops::reduce(graph(),
                            getInTensor(MeanArgGradOp::getGradInIndex()),
                            vXtoY<int64_t, std::size_t>(axes),
                            {popops::Operation::ADD},
                            prog,
                            idStr());

  // scale the grad input (reduced)
  popops::mapInPlace(graph(),
                     pe::Mul(pe::_1, pe::Const(gradOp.getScale())),
                     {out},
                     prog,
                     idStr());

  // Reshape the output, to add 1's if needed
  setOutTensor(MeanArgGradOp::getOutIndex(),
               out.reshape(outInfo(MeanArgGradOp::getOutIndex()).shape_szt()));
}

namespace {
OpxCreator<MeanOpx> meanOpxCreator({Onnx::Operators::Mean_6,
                                    Onnx::Operators::Mean_8});
OpxCreator<MeanArgGradOpx> meanGradOpxCreator(Onnx::GradOperators::MeanArgGrad);
} // namespace

} // namespace popx
} // namespace popart
