#include <poponnx/graph.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/call.hpp>
#include <poponnx/opserialiser.hpp>
#include <poponnx/scope.hpp>
#include <poponnx/tensorindex.hpp>

namespace poponnx {

CallOp::CallOp(Graph &parent_, Graph &callee_)
    : Op(Onnx::CustomOperators::Call, {parent_, ""}), callee(callee_) {
  settings.name = fmt::format("Call_{}", callee_.id);

  // set the Phase, if possible
  const auto &graphOps = callee.get().getOps();
  std::vector<Phase> graphPhases;
  graphPhases.reserve(graphOps.size());
  for (auto &op : graphOps) {
    graphPhases.push_back(op.second->getPhase());
  }

  auto allAre = [&graphPhases](Phase target) {
    return std::all_of(graphPhases.begin(),
                       graphPhases.end(),
                       [target](Phase p) { return p == target; });
  };

  if (allAre(Phase::FWD)) {
    setPhase(Phase::FWD);
  } else if (allAre(Phase::BWD)) {
    setPhase(Phase::BWD);
  }
}

void CallOp::setup() {}

std::unique_ptr<Op> CallOp::clone() const { return make_unique<CallOp>(*this); }

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
