#include <popops/ElementWise.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/negate.hpp>
#include <poponnx/popx/op/negatex.hpp>
#include <poponnx/popx/opxmanager.hpp>

namespace poponnx {
namespace popx {

NegateOpx::NegateOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<NegateOp>(op, Onnx::Operators::Neg_6);
}

void NegateOpx::grow(poplar::program::Sequence &prog) const {
  setOutTensor(0,
               popops::map(graph(),
                           popops::expr::UnaryOpType::NEGATE,
                           getInTensor(0),
                           prog,
                           idStr()));
}

NegateGradOpx::NegateGradOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<NegateGradOp>(op, Onnx::GradOperators::NegGrad);
}

void NegateGradOpx::grow(poplar::program::Sequence &prog) const {
  setOutTensor(0,
               popops::map(graph(),
                           popops::expr::UnaryOpType::NEGATE,
                           getInTensor(0),
                           prog,
                           idStr()));
}

namespace {
static OpxCreator<NegateOpx> negOpxCreator(Onnx::Operators::Neg_6);
static OpxCreator<NegateGradOpx>
    negGradOpxCreator(Onnx::GradOperators::NegGrad);
} // namespace

} // namespace popx
} // namespace poponnx
