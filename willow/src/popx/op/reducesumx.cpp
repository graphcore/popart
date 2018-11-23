#include <algorithm>
#include <iterator>
#include <vector>

#include <poponnx/error.hpp>
#include <poponnx/op/reducesum.hpp>
#include <poponnx/popx/op/reducesumx.hpp>
#include <poponnx/tensor.hpp>

#include <popops/Reduce.hpp>

namespace willow {
namespace popx {

ReduceSumOpx::ReduceSumOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (!op->isConvertibleTo<ReduceSumOp>()) {
    throw error("cannot create ReduceSumOpx from " + op->op_type());
  }
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
  if (op->opType != OpType::REDUCESUMGRAD) {
    throw error("cannot create ReduceSumGradOpx from " + op->op_type());
  }
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

} // namespace popx
} // namespace willow
