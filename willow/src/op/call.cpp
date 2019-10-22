#include <memory>
#include <popart/graph.hpp>
#include <popart/op/call.hpp>
#include <popart/opserialiser.hpp>
#include <popart/scope.hpp>
#include <popart/tensorindex.hpp>

namespace popart {

CallOp::CallOp(Graph &parent_, Graph &callee_)
    : Op(Onnx::CustomOperators::Call, {parent_, ""}), callee(callee_) {
  settings.name = logging::format("Call_{}", callee_.id);
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

VGraphId CallOp::getIntrospectionInVirtualGraphId(InIndex index) const {
  if (index > -1) {
    auto num_ids = getCalledGraph().getInputIds().size();
    if (index >= num_ids)
      throw error("[getIntrospectionInVirtualGraphId] "
                  "CallOp ({}) has {} inputs, but requested index is {}",
                  debugName(),
                  num_ids,
                  index);

    auto tensor_id = getCalledGraph().getInputId(index);
    auto tensor    = getCalledGraph().getTensors().get(tensor_id);

    // Callee introspection
    for (auto consumer : tensor->consumers.getOps()) {
      if (auto call = dynamic_cast<CallOp *>(consumer)) {
        auto subindex = consumer->input->indicesMap().at(tensor)[0];
        if (consumer->hasVirtualGraphId()) {
          // Also works if the callee is another subgraph
          auto intropId = consumer->getIntrospectionInVirtualGraphId(subindex);
          if (intropId > -1)
            return intropId;
        }
      }
    }

    // Fallback 1: The tensor knows it's own VGID
    // We ask this only after callee introspection, because otherwise the
    // CallOp's VGID will be reported, which can be wrong if it's nested
    // consuming operator is on another virtual graph.
    if (tensor->hasVirtualGraphId()) {
      // Tensor has VirtualGraphID given by it's producer or consumer
      auto vgId = tensor->getVirtualGraphId();
      if (vgId > -1) {
        return vgId;
      }
    }
  }

  // Fallback 2: No VGID determined by introspection or tensor
  return Op::hasVirtualGraphId() ? Op::getVirtualGraphId() : -1;
}

VGraphId CallOp::getIntrospectionOutVirtualGraphId(OutIndex index) const {
  if (index > -1) {
    auto num_ids = getCalledGraph().getOutputIds().size();
    if (index >= num_ids)
      throw error("[getIntrospectionOutVirtualGraphId] "
                  "CallOp ({}) has {} inputs, but requested index is {}",
                  debugName(),
                  num_ids,
                  index);

    auto tensor_id = getCalledGraph().getOutputId(index);
    auto tensor    = getCalledGraph().getTensors().get(tensor_id);

    // Callee introspection
    auto producer = tensor->getProducer();
    if (auto call = dynamic_cast<CallOp *>(producer)) {
      auto subindex = producer->output->indicesMap().at(tensor)[0];
      if (producer->hasVirtualGraphId()) {
        // Also works if the callee is another subgraph
        auto vgId = producer->getIntrospectionOutVirtualGraphId(subindex);
        if (vgId > -1) {
          return vgId;
        }
      }
    }

    // Fallback 1: The tensor knows it's own VGID
    // We ask this only after callee introspection, because otherwise the
    // CallOp's VGID will be reported, which can be wrong if it's nested
    // consuming operator is on another virtual graph.
    if (tensor->hasVirtualGraphId()) {
      // Tensor has VirtualGraphID given by it's producer or consumer
      auto vgId = tensor->getVirtualGraphId();
      if (vgId > -1) {
        return vgId;
      }
    }
  }

  // Fallback 2: No VGID determined by introspection or tensor
  return Op::hasVirtualGraphId() ? Op::getVirtualGraphId() : -1;
}

} // namespace popart
