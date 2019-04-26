#include <poponnx/error.hpp>
#include <poponnx/graph.hpp>
#include <poponnx/names.hpp>
#include <poponnx/op.hpp>
#include <poponnx/tensor.hpp>

#include <poponnx/transforms/virtual_graph_check.hpp>

namespace poponnx {

std::size_t VirtualGraphCheck::id() {
  return typeid(VirtualGraphCheck).hash_code();
}

bool VirtualGraphCheck::apply(Graph &graph) const {
  if (graph.getOps().size() == 0) {
    return true;
  }

  auto *first_op    = graph.getOps().begin()->second.get();
  bool has_sharding = static_cast<bool>(first_op->getVirtualGraphId());

  for (auto &id_op : graph.getOps()) {
    Op *op       = id_op.second.get();
    bool sharded = static_cast<bool>(op->getVirtualGraphId());
    if (sharded != has_sharding) {
      throw error("{} {} virtual graph attribute but {} {}",
                  first_op->debugName(),
                  has_sharding ? "has" : "does not have",
                  op->debugName(),
                  sharded ? "does" : "does not");
    }
  }

  return true;
}

namespace {
bool init = Transform::registerTransform(new VirtualGraphCheck);
}

} // namespace poponnx
