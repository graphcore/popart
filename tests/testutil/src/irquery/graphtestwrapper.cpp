// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <testutil/irquery/graphtestwrapper.hpp>

namespace popart {
namespace irquery {

GraphTestWrapper::GraphTestWrapper(Ir &ir_, const GraphId &id)
    : TestWrapper<std::reference_wrapper<Graph>>{ir_, ir_.getGraph(id)} {}

OpsTestWrapper GraphTestWrapper::ops() {
  Graph &graph = wrappedObj;
  std::vector<Op *> ops;
  for (auto &entry : graph.getOps()) {
    ops.push_back(entry.second.get());
  }
  return OpsTestWrapper{ir, ops, graph.getGraphString()};
}

TensorIndexMapTestWrapper GraphTestWrapper::inputs() {
  Graph &graph = wrappedObj;
  std::map<int, Tensor *> tensorIndexMap;

  int index = 0;
  for (auto &id : graph.getInputIds()) {
    tensorIndexMap[index++] = graph.getTensors().get(id);
  }

  return TensorIndexMapTestWrapper{
      ir, tensorIndexMap, graph.getGraphString(), "input", "inputs"};
}

TensorIndexMapTestWrapper GraphTestWrapper::outputs() {
  Graph &graph = wrappedObj;
  std::map<int, Tensor *> tensorIndexMap;

  int index = 0;
  for (auto &id : graph.getOutputIds()) {
    tensorIndexMap[index++] = graph.getTensors().get(id);
  }

  return TensorIndexMapTestWrapper{
      ir, tensorIndexMap, graph.getGraphString(), "output", "outputs"};
}

} // namespace irquery
} // namespace popart
