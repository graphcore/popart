#include <algorithm>
#include <iterator>
#include <vector>

#include <poponnx/error.hpp>
#include <poponnx/op/reducesum.hpp>
#include <poponnx/popx/op/reducesumx.hpp>
#include <poponnx/popx/opxmanager.hpp>
#include <poponnx/tensor.hpp>

#include <popops/Reduce.hpp>

namespace poponnx {
namespace popx {

ReduceSumOpx::ReduceSumOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<ReduceSumOp>(op);
}

template <typename T1, typename T2>
static std::vector<T1> vector_cast(const std::vector<T2> &xs) {
  std::vector<T1> ys;

  ys.reserve(xs.size());
  for (const auto &x : xs) {
    ys.emplace_back(static_cast<T1>(x));
  }

  return ys;
}

void ReduceSumOpx::grow(poplar::program::Sequence &prog) const {
  const auto op    = dynamic_cast<ReduceSumOp *>(op_p);
  const auto input = get(inId(0));

  auto output_tensor = popops::reduce(graph(),
                                      input,
                                      vector_cast<std::size_t>(op->getAxes()),
                                      {popops::Operation::ADD},
                                      prog);

  insert(outId(0), output_tensor.reshape(outInfo(0).shape_szt()));
}

ReduceSumGradOpx::ReduceSumGradOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<ReduceSumGradOp>(op, Onnx::GradOperators::ReduceSumGrad);
}

void ReduceSumGradOpx::grow(poplar::program::Sequence &prog) const {
  const auto op        = dynamic_cast<ReduceSumGradOp *>(op_p);
  auto output          = cloneNcopy(prog, inId(0));
  auto input_shape     = inShape(0);
  auto output_shape    = outShape(0);
  const auto new_shape = vector_cast<std::size_t>(op->backwardShape());

  output = output.reshape(new_shape);

  // Broadcasting across each dimension
  for (int dim = 0; dim < new_shape.size(); ++dim) {
    if (new_shape[dim] != output_shape[dim]) {
      output = output.broadcast(static_cast<uint32_t>(output_shape[dim]), dim);
    }
  }

  // output now matches the shape of output_shape
  insert(outId(0), output);
}

namespace {
OpxCreator<ReduceSumOpx> reduceSumOpxCreator(Onnx::Operators::ReduceSum_1);
OpxCreator<ReduceSumGradOpx>
    reduceSumGradGradOpxCreator(Onnx::GradOperators::ReduceSumGrad);
} // namespace

} // namespace popx
} // namespace poponnx
