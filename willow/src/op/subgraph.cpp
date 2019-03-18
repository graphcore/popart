
#include <poponnx/op/subgraph.hpp>
#include <poponnx/opserialiser.hpp>

namespace poponnx {

SubgraphOp::SubgraphOp(Ir &ir_, int64_t cacheId_)
    : Op(Onnx::CustomOperators::Subgraph, {ir_, ""}), cacheId(cacheId_) {}

std::vector<Op *> SubgraphOp::getOps() {
  std::vector<Op *> o;
  for (auto &_o : getChildOpsInfo()) {
    o.push_back(_o.op);
  }
  return o;
}

void SubgraphOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("cacheId", cacheId);
}

} // namespace poponnx
