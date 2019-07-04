#include <memory>
#include <poponnx/graph.hpp>
#include <poponnx/op/call.hpp>
#include <poponnx/opserialiser.hpp>
#include <poponnx/scope.hpp>
#include <poponnx/tensorindex.hpp>

namespace poponnx {

CallOp::CallOp(Graph &parent_, Graph &callee_)
    : Op(Onnx::CustomOperators::Call, {parent_, ""}), callee(callee_) {
  settings.name = fmt::format("Call_{}", callee_.id);

  // Set the position w.r.t loss, if possible. If any of the internal ops
  // is connected to the final loss, then so is this CallOp.
  const auto &graphOps = callee.get().getOps();
  for (auto &id_op : graphOps) {
    auto op = id_op.second.get();
    if (op->toLoss == PathToLoss::Yes) {
      toLoss = PathToLoss::Yes;
    }
    if (op->fromLoss == PathFromLoss::Yes) {
      fromLoss = PathFromLoss::Yes;
    }
  }
}

void CallOp::setup() {}

std::unique_ptr<Op> CallOp::clone() const {
  return std::make_unique<CallOp>(*this);
}

const Graph &CallOp::getCalledGraph() const { return callee.get(); }

void CallOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("callee", callee.get().id.str());
}

bool CallOp::isInputModified(InIndex index) {
  auto tensor_id = getCalledGraph().getInputId(index);
  auto tensor    = getCalledGraph().getTensors().get(tensor_id);

  for (auto consumer : tensor->consumers.getOps()) {
    for (auto i : consumer->input->indices(tensor)) {
      if (consumer->aliases(i).isEmpty() == false ||
          consumer->modifies(i).isEmpty() == false) {
        return true;
      }
    }
  }

  return false;
}

std::vector<const Graph *> CallOp::getCalledGraphs() const {
  return {&getCalledGraph()};
}

std::vector<TensorId> CallOp::getInputsForGraph(const Graph &) const {
  std::vector<TensorId> result;
  for (int i = 0; i < input->n(); i++) {
    result.push_back(inId(i));
  }
  return result;
}

} // namespace poponnx
