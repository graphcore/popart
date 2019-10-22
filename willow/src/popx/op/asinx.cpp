#include <iterator>
#include <vector>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/op/asin.hpp>
#include <popart/popx/op/asinx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensorindex.hpp>

#include <popops/ElementWise.hpp>

namespace pe = popops::expr;

namespace popart {
namespace popx {

AsinInplaceOpx::AsinInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryInplaceOpx(op, devicex, AsinComputex::get()) {
  verifyOp<AsinInplaceOp>(op, Onnx::CustomOperators::AsinInplace);
}

AsinOpx::AsinOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOutplaceOpx(op, devicex, AsinComputex::get()) {
  verifyOp<AsinOp>(op, Onnx::Operators::Asin_7);
}

poplar::Tensor AsinComputex::outplace(poplar::program::Sequence &p,
                                      poplar::Graph &g,
                                      const poplar::Tensor &t,
                                      const std::string &s) const {
  auto outTensor = cloneNcopy(p, g, t);
  inplace(p, g, outTensor, s);
  return outTensor;
}

void AsinComputex::inplace(poplar::program::Sequence &p,
                           poplar::Graph &g,
                           const poplar::Tensor &t,
                           const std::string &s) const {

  popops::mapInPlace(g, popops::expr::UnaryOpType::ASIN, t, p, s);
}

AsinGradOpx::AsinGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<AsinGradOp>(op, Onnx::GradOperators::AsinGrad);
}

void AsinGradOpx::grow(poplar::program::Sequence &prog) const {
  auto op              = getOp<AsinGradOp>();
  const auto input     = getInTensor(AsinGradOp::getGradInIndex());
  const auto fwd_input = getInTensor(AsinGradOp::getFwdArgInIndex());

  // The derivative of the asin function can be constructed from normal
  // functions d/dx asin(x) = 1/sqrt(1-x^2)
  std::vector<std::unique_ptr<popops::expr::Expr>> exprs;
  exprs.push_back(
      std::make_unique<pe::Sub>(pe::Const(1.0f), pe::Mul(pe::_2, pe::_2)));
  exprs.push_back(std::make_unique<pe::Sqrt>(*exprs.back()));
  exprs.push_back(std::make_unique<pe::Divide>(pe::Const(1.0f), *exprs.back()));
  exprs.push_back(std::make_unique<pe::Mul>(pe::_1, *exprs.back()));

  auto output = popops::map(graph(),
                            *exprs.back(),
                            {input, fwd_input},
                            prog,
                            debugPrefix("inverse_sine_grad"));

  setOutTensor(AsinGradOp::getOutIndex(), output);
}

namespace {
OpxCreator<AsinOpx> asinOpxCreator(Onnx::Operators::Asin_7);
OpxCreator<AsinInplaceOpx>
    asinInplaceOpxCreator(Onnx::CustomOperators::AsinInplace);
OpxCreator<AsinGradOpx> asinGradOpxCreator(Onnx::GradOperators::AsinGrad);
} // namespace

} // namespace popx
} // namespace popart
