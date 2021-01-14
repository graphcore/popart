// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <onnx/onnx_pb.h>
#include <onnxpasses/nodepatterns/binary_ops.hpp>
#include <onnxpasses/nodepatterns/conv.hpp>
#include <onnxpasses/nodepatterns/spacedepth.hpp>
#include <onnxpasses/nodepatterns/trig.hpp>
#include <onnxpasses/onnxtoonnx.hpp>
#include <onnxpasses/patterntarget.hpp>
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

  auto target = std::make_shared<PatternTarget>(g);

  std::vector<std::unique_ptr<NodePattern>> patterns;
  patterns.push_back(std::make_unique<ConvWithBias>(target));
  patterns.push_back(std::make_unique<MultiConvWithBias>(target));
  patterns.push_back(std::make_unique<Tan>(target));
  patterns.push_back(std::make_unique<Asinh>(target));
  patterns.push_back(std::make_unique<Acosh>(target));
  patterns.push_back(std::make_unique<Atanh>(target));
  patterns.push_back(std::make_unique<Acos>(target));
  patterns.push_back(std::make_unique<Remainder>(target));
  patterns.push_back(std::make_unique<Mod>(target));
  patterns.push_back(std::make_unique<DepthToSpace>(target));
  patterns.push_back(std::make_unique<SpaceToDepth>(target));

  /**
   * The ONNX spec ensures that the Nodes appear in topological order.
   *
   * We iterate through the Nodes in the original Graph in topological order,
   * and try matching the NodePatterns to them. If a NodePattern matches, then
   * we immediately process it and proceed to the next Node. If no NodePatterns
   * match, then a copy of the Node is inserted in the new list (schedule).
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

    // No patterns can transform this Node, so it enters the the Graph
    // unchanged.
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
