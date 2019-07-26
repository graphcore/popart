#include <popnn/Loss.hpp>

#include <popart/error.hpp>
#include <popart/op/argmax.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/argmaxx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

poplar::Tensor ArgMaxOpx::extremaOp(poplar::program::Sequence &prog,
                                    const poplar::Tensor &input) const {
  return popnn::argMax(graph(), input, prog, idStr());
}

namespace {
OpxCreator<ArgMaxOpx> argMaxOpxCreator(Onnx::Operators::ArgMax_1);
} // namespace

} // namespace popx
} // namespace popart
