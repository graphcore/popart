#include <popnn/Loss.hpp>

#include <popart/op/argmin.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/argminx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

poplar::Tensor ArgMinOpx::extremaOp(poplar::program::Sequence &prog,
                                    const poplar::Tensor &input) const {
  return popnn::argMin(graph(), input, prog, debugPrefix("argmin"));
}

namespace {
OpxCreator<ArgMinOpx> argMinOpxCreator({Onnx::Operators::ArgMin_1,
                                        Onnx::Operators::ArgMin_11});
} // namespace

} // namespace popx
} // namespace popart
