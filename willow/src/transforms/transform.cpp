#include <memory>
#include <popart/logging.hpp>
#include <popart/transforms/transform.hpp>

namespace popart {

// Map from the transform Id to a transform
using TransformMap = std::map<std::size_t, std::unique_ptr<Transform>>;

static TransformMap &getTransformMap() {
  static TransformMap transform_map;
  return transform_map;
}

void Transform::applyTransform(std::size_t transformId, Graph &graph) {
  auto &transform = getTransformMap().at(transformId);
  logging::transform::info("Applying Graph transform {}", transform->getName());
  transform->apply(graph);
}

bool Transform::registerTransform(Transform *transform) {
  getTransformMap().emplace(transform->getId(), transform);
  return true;
}

} // namespace popart
