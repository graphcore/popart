// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <onnx/onnx_pb.h>
#include <onnxpasses/nodepatterns.hpp>
#include <onnxpasses/onnxtoonnx.hpp>
#include <onnxpasses/suffixer.hpp>
#include <string>
#include <popart/error.hpp>

namespace popart {
namespace onnxpasses {

IOnnxToOnnx::IOnnxToOnnx()  = default;
IOnnxToOnnx::~IOnnxToOnnx() = default;

Canonnxalizer::Canonnxalizer()  = default;
Canonnxalizer::~Canonnxalizer() = default;

using namespace ONNX_NAMESPACE;

void Canonnxalizer::canonnxalize(GraphProto &g) const {

  Suffixer suffixer(g);

  std::vector<std::unique_ptr<NodePattern>> patterns;
  patterns.push_back(std::make_unique<ConvWithBias>(g, suffixer));
  patterns.push_back(std::make_unique<MultiConvWithBias>(g, suffixer));
  patterns.push_back(std::make_unique<Tan>(g, suffixer));
  patterns.push_back(std::make_unique<Asinh>(g, suffixer));
  patterns.push_back(std::make_unique<Acosh>(g, suffixer));
  patterns.push_back(std::make_unique<Atanh>(g, suffixer));
  patterns.push_back(std::make_unique<Acos>(g, suffixer));

  /**
   * The ONNX spec ensures that the Nodes appear in topological order.
   *
   * We iterate theough the Nodes in the original Graph in topological order,
   * and try matching the NodePatterns to them. If a NodePattern matches, then
   * we immediately proceed to the next Node. If no NodePatterns match, then a
   * copy of the Node is inserted in the new list (schedule).
   * */
  auto prevNodes = g.node();
  g.mutable_node()->Clear();

  for (const auto &node : prevNodes) {
    bool nodeMatch{false};
    for (const auto &p : patterns) {
      if (p->run(node)) {
        nodeMatch = true;
        break;
      }
    }
    if (!nodeMatch) {
      auto nxt = g.mutable_node()->Add();
      *nxt     = node;
    }
  }
}

GraphProto IOnnxToOnnx::getCanonnxalized(const GraphProto &inputGraph) const {
  // Create a copy of the input Graph, and modify inplace.
  auto g = inputGraph;
  canonnxalize(g);
  return g;
}

} // namespace onnxpasses
} // namespace popart
