#include <memory>
#include <popart/error.hpp>
#include <popart/op/onehot.hpp>
#include <popart/popx/op/onehotx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/util.hpp>

#include <popops/ElementWise.hpp>
#include <popops/Encoding.hpp>
#include <popops/Expr.hpp>

#include <queue>
#include <popops/Reduce.hpp>

namespace pe = popops::expr;

namespace popart {
namespace popx {

OnehotOpx::OnehotOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<OnehotOpx>(op,
                      {Onnx::Operators::OneHot_9, Onnx::Operators::OneHot_11});
}

void OnehotOpx::grow(poplar::program::Sequence &prog) const {

  OnehotOp &onehotOp = getOp<OnehotOp>();

  const poplar::Tensor &indices = getInTensor(OnehotOp::getIndicesInIndex());
  const poplar::Tensor &values  = getInTensor(OnehotOp::getValuesInIndex());

  // Create a new output tensor with the type of the values
  const auto shape = vXtoY<int64_t, std::size_t>(
      onehotOp.outInfo(OnehotOp::getOutIndex()).shape());
  auto output =
      graph().addVariable(values.elementType(), shape, debugPrefix("output"));

  // roll the one-hot dimension to the end, if needed
  if (onehotOp.getAxis() != -1) {
    output = output.dimRoll(static_cast<unsigned>(onehotOp.getAxis()),
                            output.rank() - 1);
  }

  // flatten all but one-hot dimension so we are left with a 2d tensor
  output = output.reshapePartial(0, output.rank() - 1, {indices.numElements()});

  // call popops function to generate the one-hot matrix
  popops::encodeOneHot(
      graph(), indices.flatten(), output, prog, debugPrefix("onehot"));

  // The "owner" of all expr nodes:
  std::vector<std::unique_ptr<popops::expr::Expr>> exprs;

  // a = output, b = values.slice({0, 1}, 0), c = values.slice({1, 2}, 0)
  // First append a * c, we use this later
  exprs.push_back(std::make_unique<pe::Mul>(pe::_1, pe::_3));
  // Append a - 1
  exprs.push_back(std::make_unique<pe::Sub>(pe::_1, pe::Const(1)));
  // ...then negate
  exprs.push_back(std::make_unique<pe::Neg>(*exprs.back()));
  // ...then multiply by b
  exprs.push_back(std::make_unique<pe::Mul>(*exprs.back(), pe::_2));
  // then multiply this by a * c which we appended first
  exprs.push_back(std::make_unique<pe::Add>(*exprs.front(), *exprs.back()));

  // Apply the above expression to the input tensors
  popops::mapInPlace(graph(),
                     *exprs.back(),
                     {output, values.slice({0, 1}, 0), values.slice({1, 2}, 0)},
                     prog,
                     debugPrefix("combine"));

  // reshape the flattened output dimensions back to their original shape
  output = output.reshapePartial(0, 1, indices.shape());

  // roll the one-hot dimension back to axis, if needed
  if (onehotOp.getAxis() != -1) {
    output = output.dimRoll(output.rank() - 1,
                            static_cast<unsigned>(onehotOp.getAxis()));
  }

  // Done
  setOutTensor(OnehotOp::getOutIndex(), output);
}

OnehotGradOpx::OnehotGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<OnehotGradOpx>(op, Onnx::GradOperators::OneHotGrad);
}

void OnehotGradOpx::grow(poplar::program::Sequence &prog) const {

  OnehotGradOp &onehotGradOp = getOp<OnehotGradOp>();

  const poplar::Tensor &indices =
      getInTensor(OnehotGradOp::getIndicesInIndex());
  poplar::Tensor gradInput = getInTensor(OnehotGradOp::getGradInIndex());

  // roll the one-hot dimension to the end, if needed
  if (onehotGradOp.getAxis() != -1) {
    gradInput = gradInput.dimRoll(static_cast<unsigned>(onehotGradOp.getAxis()),
                                  gradInput.rank() - 1);
  }

  // flatten all but one-hot axis
  gradInput = gradInput.reshapePartial(
      0, gradInput.rank() - 1, {indices.numElements()});

  // Create a new output tensor with the type of the values
  auto mask = graph().addVariable(gradInput.elementType(),
                                  gradInput.shape(),
                                  poplar::VariableMappingMethod::LINEAR,
                                  debugPrefix("mask"));

  // Call popops function to generate the one-hot matrix mask
  popops::encodeOneHot(
      graph(), indices.flatten(), mask, prog, debugPrefix("onehot"));

  auto hotMask = popops::map(graph(),
                             pe::Mul(pe::_1, pe::_2),
                             {gradInput, mask},
                             prog,
                             debugPrefix("hotMask"));

  auto hotValue = popops::reduce(graph(),
                                 hotMask.flatten(),
                                 {0},
                                 {popops::Operation::ADD},
                                 prog,
                                 debugPrefix("hotValue"));

  auto nothotMask =
      popops::map(graph(),
                  pe::Mul(pe::Neg(pe::Sub(pe::_1, pe::Const(1))), pe::_2),
                  {mask, gradInput},
                  prog,
                  debugPrefix("nothotMask"));

  auto nothotValue = popops::reduce(graph(),
                                    nothotMask.flatten(),
                                    {0},
                                    {popops::Operation::ADD},
                                    prog,
                                    debugPrefix("nothotValue"));

  const auto shape = vXtoY<int64_t, std::size_t>(onehotGradOp.getOutputShape());

  // Create a new output tensor
  auto output = graph().addVariable(
      gradInput.elementType(), shape, debugPrefix("output"));

  // The output.slice method returns a view on the underling output tensor
  // that we can write the hot or not hot value into

  popops::mapInPlace(graph(),
                     popops::expr::BinaryOpType::ADD,
                     output.slice({0, 1}, 0),
                     nothotValue,
                     prog,
                     debugPrefix("addNothot"));

  popops::mapInPlace(graph(),
                     popops::expr::BinaryOpType::ADD,
                     output.slice({1, 2}, 0),
                     hotValue,
                     prog,
                     debugPrefix("addHot"));

  setOutTensor(OnehotGradOp::getOutIndex(), output);
}

namespace {
OpxCreator<OnehotOpx> onehotOpxCreator({Onnx::Operators::OneHot_9,
                                        Onnx::Operators::OneHot_11});
OpxCreator<OnehotGradOpx> onehotGradOpxCreator(Onnx::GradOperators::OneHotGrad);
} // namespace

} // namespace popx
} // namespace popart
