#include <popnn/Loss.hpp>

#include <poponnx/op/argmin.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/op/argminx.hpp>
#include <poponnx/popx/opxmanager.hpp>

namespace poponnx {
namespace popx {

poplar::Tensor ArgMinOpx::extremaOp(poplar::program::Sequence &prog,
                                    const poplar::Tensor &input) const {
  return popnn::argMin(graph(), input, prog, idStr());
}

namespace {
OpxCreator<ArgMinOpx> argMinOpxCreator(Onnx::Operators::ArgMin_1);
} // namespace

} // namespace popx
} // namespace poponnx
