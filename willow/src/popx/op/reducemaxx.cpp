#include <algorithm>
#include <iterator>
#include <vector>

#include <popart/error.hpp>
#include <popart/op/reducemax.hpp>
#include <popart/popx/op/reducemaxx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/util.hpp>

#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>

namespace pe = popops::expr;

namespace popart {
namespace popx {

ReduceMaxOpx::ReduceMaxOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<ReduceMaxOp>(op);
}

void ReduceMaxOpx::grow(poplar::program::Sequence &prog) const {
  const auto op    = getOp<ReduceMaxOp>();
  const auto input = getInTensor(ReduceMaxOp::getInIndex());

  auto output_tensor = popops::reduce(graph(),
                                      input,
                                      vector_cast<std::size_t>(op.getAxes()),
                                      {popops::Operation::MAX},
                                      prog,
                                      debugPrefix("max"));

  setOutTensor(
      ReduceMaxOp::getOutIndex(),
      output_tensor.reshape(outInfo(ReduceMaxOp::getOutIndex()).shape_szt()));
}

ReduceMaxGradOpx::ReduceMaxGradOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<ReduceMaxGradOp>(op, Onnx::GradOperators::ReduceMaxGrad);
}

void ReduceMaxGradOpx::grow(poplar::program::Sequence &prog) const {
  const auto op = getOp<ReduceMaxGradOp>();
  auto output   = cloneNcopy(prog, getInTensor(ReduceMaxGradOp::getInIndex()));
  auto mask =
      cloneNcopy(prog, getInTensor(ReduceMaxGradOp::getFwdOutInIndex()));
  auto input_shape     = inShape(ReduceMaxGradOp::getInIndex());
  auto output_shape    = outShape(ReduceMaxGradOp::getOutIndex());
  const auto new_shape = vector_cast<std::size_t>(op.backwardShape());

  output = output.reshape(new_shape);
  mask   = mask.reshape(new_shape);

  // Broadcasting across each dimension
  for (int dim = 0; dim < new_shape.size(); ++dim) {
    if (new_shape[dim] != output_shape[dim]) {
      output = output.broadcast(static_cast<uint32_t>(output_shape[dim]), dim);
      mask   = mask.broadcast(static_cast<uint32_t>(output_shape[dim]), dim);
    }
  }

  output = popops::map(
      graph(),
      pe::Mul(pe::Add(pe::Signum(pe::Sub(pe::_2, pe::_1)), pe::Const(1)),
              pe::_3),
      {mask, getInTensor(ReduceMaxGradOp::getFwdInInIndex()), output},
      prog,
      debugPrefix("maskmul"));

  // output now matches the shape of output_shape
  setOutTensor(ReduceMaxGradOp::getOutIndex(), output);
}

namespace {
OpxCreator<ReduceMaxOpx> reduceMaxOpxCreator(Onnx::Operators::ReduceMax_1);
OpxCreator<ReduceMaxGradOpx>
    reduceMaxGradGradOpxCreator(Onnx::GradOperators::ReduceMaxGrad);
} // namespace

} // namespace popx
} // namespace popart
