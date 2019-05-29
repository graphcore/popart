#include <popnn/Loss.hpp>

#include <poponnx/error.hpp>
#include <poponnx/op/argmax.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/op/argmaxx.hpp>
#include <poponnx/popx/opxmanager.hpp>

namespace poponnx {
namespace popx {

poplar::Tensor ArgMaxOpx::extremaOp(poplar::program::Sequence &prog,
                                    const poplar::Tensor &input) const {
  return popnn::argMax(graph(), input, prog, idStr());
}

namespace {
OpxCreator<ArgMaxOpx> argMaxOpxCreator(Onnx::Operators::ArgMax_1);
} // namespace

} // namespace popx
} // namespace poponnx
