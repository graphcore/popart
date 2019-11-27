#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/recomputeprereq.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/recomputeprereqx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

namespace {
OpxCreator<RecomputePrereqOpx>
    recomputePrereqOpxCreator(Onnx::CustomOperators::RecomputePrereq);
}
} // namespace popx
} // namespace popart
