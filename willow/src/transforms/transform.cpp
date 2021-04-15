// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/logging.hpp>
#include <popart/transforms/transform.hpp>

#include <poprithms/logging/timepartitionlogger.hpp>

namespace popart {

namespace {

// Map from the transform Id to a transform
using TransformMap = std::map<std::size_t, std::unique_ptr<Transform>>;

static TransformMap &getTransformMap() {
  static TransformMap transform_map;
  return transform_map;
}

} // anonymous namespace

void Transform::applyTransform(std::size_t transformId, Graph &graph) {
  auto &transform = getTransformMap().at(transformId);
  Transform::applyTransformHelper<Transform, Graph &>(
      *transform, graph.getIr(), std::ref(graph));
}

bool Transform::registerTransform(Transform *transform) {
  getTransformMap().emplace(transform->getId(), transform);
  return true;
}

void Transform::startStopwatch(Ir &ir) {
  const auto name = getName();
  auto &logger    = ir.timePartitionLogger();
  logger.start(name);
}

void Transform::stopStopwatch(Ir &ir) {
  auto &logger = ir.timePartitionLogger();
  logger.stop();
}

} // namespace popart
