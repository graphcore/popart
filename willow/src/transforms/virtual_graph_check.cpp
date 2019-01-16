#include <poponnx/error.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/names.hpp>
#include <poponnx/op.hpp>
#include <poponnx/tensor.hpp>

#include <poponnx/transforms/virtual_graph_check.hpp>

namespace poponnx {

std::size_t VirtualGraphCheck::id() {
  return typeid(VirtualGraphCheck).hash_code();
}

bool VirtualGraphCheck::apply(Ir &ir) const {
  if (ir.getOps().size() == 0) {
    return true;
  }

  auto *first_op    = ir.getOps().begin()->second.get();
  bool has_sharding = first_op->nAtts.hasAttribute(sVirtualGraphAttribute);

  for (auto &id_op : ir.getOps()) {
    Op *op       = id_op.second.get();
    bool sharded = op->nAtts.hasAttribute(sVirtualGraphAttribute);
    if (sharded != has_sharding) {
      throw error("{} has different virtual graph attribute to {}",
                  first_op->debugName(),
                  op->debugName());
    }
  }

  return true;
}

namespace {
bool init = Transform::registerTransform(new VirtualGraphCheck);
}

} // namespace poponnx
