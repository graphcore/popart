// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <cstddef>
#include <map>
#include <memory>
#include <poprithms/logging/timepartitionlogger.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/logging.hpp>
#include <popart/transforms/transform.hpp>
#include <poparttracepoint.hpp>

namespace popart {

// Map from the transform Id to a transform
using TransformMap = std::map<std::size_t, std::unique_ptr<Transform>>;

static TransformMap &getTransformMap() {
  static TransformMap transform_map;
  return transform_map;
}

void Transform::applyTransform(std::size_t transformId, Graph &graph) {

  auto &transform = getTransformMap().at(transformId);

  const auto scopedTimer =
      graph.getIr().timePartitionLogger().scopedStopwatch(transform->getName());

  PopartTracepoint tp(
      logging::format("Applying transform '{}'", transform->getName()));
  logging::transform::info("Applying Graph transform {}", transform->getName());
  transform->apply(graph);
}

bool Transform::registerTransform(Transform *transform) {
  getTransformMap().emplace(transform->getId(), transform);
  return true;
}

} // namespace popart
