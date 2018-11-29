#include <poponnx/logging.hpp>
#include <poponnx/transforms/transform.hpp>

namespace poponnx {

// Map from the transform Id to a transform
using TransformMap = std::map<std::size_t, std::unique_ptr<Transform>>;

static TransformMap &getTransformMap() {
  static TransformMap transform_map;
  return transform_map;
}

void Transform::applyTransform(std::size_t transformId, Ir &ir) {
  auto &transform = getTransformMap().at(transformId);
  logging::transform::debug("Applying transform {}", transform->getName());
  transform->apply(ir);
}

bool Transform::registerTransform(Transform *transform) {
  getTransformMap().emplace(transform->getId(), transform);
  return true;
}

} // namespace poponnx
