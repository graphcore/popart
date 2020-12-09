#include <onnxpasses/patterntarget.hpp>
#include <string>
#include <poprithms/ndarray/shape.hpp>

namespace popart {
namespace onnxpasses {

PatternTarget::PatternTarget(GraphProto &g_)
    : g(g_), nodes(g.mutable_node()), suffixer(g_) {
  // here, we will populate shapes map T31464.
}

} // namespace onnxpasses
} // namespace popart
