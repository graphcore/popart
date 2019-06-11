#include <algorithm>
#include <iterator>
#include <vector>

#include <poponnx/error.hpp>
#include <poponnx/op/reducelogsumexp.hpp>
#include <poponnx/popx/op/reducelogsumexpx.hpp>
#include <poponnx/popx/opxmanager.hpp>
#include <poponnx/tensor.hpp>

#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>

namespace pe = popops::expr;

namespace poponnx {
namespace popx {

ReduceLogSumExpOpx::ReduceLogSumExpOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<ReduceLogSumExpOp>(op);
}

void ReduceLogSumExpOpx::grow(poplar::program::Sequence &prog) const {
  const auto op        = getOp<ReduceLogSumExpOp>();
  const auto input     = getInTensor(ReduceLogSumExpOp::getInIndex());
  const auto new_shape = vector_cast<std::size_t>(op.backwardShape());

  auto maxval           = popops::reduce(graph(),
                               input,
                               vector_cast<std::size_t>(op.getAxes()),
                               {popops::Operation::MAX},
                               prog);
  auto broadcast_maxval = maxval.reshape(new_shape);

  // Broadcasting across each dimension
  for (int dim = 0; dim < new_shape.size(); ++dim) {
    if (new_shape[dim] != input.shape()[dim]) {
      broadcast_maxval = broadcast_maxval.broadcast(
          static_cast<uint32_t>(input.shape()[dim]), dim);
    }
  }

  auto expinput = popops::map(graph(),
                              pe::Exp(pe::Sub(pe::_1, pe::_2)),
                              {input, broadcast_maxval},
                              prog);

  auto output_tensor = popops::reduce(graph(),
                                      expinput,
                                      vector_cast<std::size_t>(op.getAxes()),
                                      {popops::Operation::ADD},
                                      prog);
  output_tensor      = popops::map(
      graph(), pe::Add(pe::Log(pe::_1), pe::_2), {output_tensor, maxval}, prog);

  setOutTensor(ReduceLogSumExpOp::getOutIndex(),
               output_tensor.reshape(
                   outInfo(ReduceLogSumExpOp::getOutIndex()).shape_szt()));
}

ReduceLogSumExpGradOpx::ReduceLogSumExpGradOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<ReduceLogSumExpGradOp>(op, Onnx::GradOperators::ReduceLogSumExpGrad);
}

void ReduceLogSumExpGradOpx::grow(poplar::program::Sequence &prog) const {
  const auto op        = getOp<ReduceLogSumExpGradOp>();
  auto output          = getInTensor(ReduceLogSumExpGradOp::getInIndex());
  auto scale           = getInTensor(ReduceLogSumExpGradOp::getFwdOutInIndex());
  auto fwd_input       = getInTensor(ReduceLogSumExpGradOp::getFwdInInIndex());
  auto input_shape     = inShape(ReduceLogSumExpGradOp::getInIndex());
  auto output_shape    = outShape(ReduceLogSumExpGradOp::getOutIndex());
  const auto new_shape = vector_cast<std::size_t>(op.backwardShape());

  output = output.reshape(new_shape);
  scale  = scale.reshape(new_shape);

  auto maxval           = popops::reduce(graph(),
                               fwd_input,
                               vector_cast<std::size_t>(op.getAxes()),
                               {popops::Operation::MAX},
                               prog);
  auto broadcast_maxval = maxval.reshape(new_shape);

  // Broadcasting across each dimension
  for (int dim = 0; dim < new_shape.size(); ++dim) {
    if (new_shape[dim] != output_shape[dim]) {
      output = output.broadcast(static_cast<uint32_t>(output_shape[dim]), dim);
      scale  = scale.broadcast(static_cast<uint32_t>(output_shape[dim]), dim);
      broadcast_maxval = broadcast_maxval.broadcast(
          static_cast<uint32_t>(output_shape[dim]), dim);
    }
  }

  output =
      popops::map(graph(),
                  pe::Mul(pe::Divide(pe::_1, pe::Exp(pe::Sub(pe::_2, pe::_4))),
                          pe::Exp(pe::Sub(pe::_3, pe::_4))),
                  {output, scale, fwd_input, broadcast_maxval},
                  prog);

  // output now matches the shape of output_shape
  setOutTensor(ReduceLogSumExpGradOp::getOutIndex(), output);
}

namespace {
OpxCreator<ReduceLogSumExpOpx>
    reduceLogSumExpOpxCreator(Onnx::Operators::ReduceLogSumExp_1);
OpxCreator<ReduceLogSumExpGradOpx>
    reduceLogSumExpGradGradOpxCreator(Onnx::GradOperators::ReduceLogSumExpGrad);
} // namespace

} // namespace popx
} // namespace poponnx
