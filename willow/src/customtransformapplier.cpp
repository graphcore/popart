// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <cstddef>
#include <customtransformapplier.hpp>
#include <map>
#include <memory>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/logging.hpp>

#include "popart/transforms/transform.hpp"
namespace popart {

bool CustomTransformApplier::isEnabledtransform(const std::string &name) {
  // Transforms is available by default.
  return true;
}

void CustomTransformApplier::applyCustomTransforms(
    const std::string &checkPoint) {
  // Experimental for inference first.
  if (ir.get().isTesting()) {
    std::set<std::string> validOptions = {
        "Fwd0", "Fwd1", "Bwd0", "PreAlias", "MainLoops", "Final"};
    for (const auto &key_value :
         ir.get()
             .getSessionOptions()
             .experimentalSettings.customTransformApplierSettings) {
      const std::string &key = key_value.first;
      if (validOptions.find(key) == validOptions.end()) {
        logging::transform::err(
            "'{}' is not a valid customTransformApplierSettings option.", key);
      }

      if (key == checkPoint) {
        for (const auto &transformName : key_value.second) {
          auto transformId = Transform::getIdFromName(transformName);
          if (transformId != 0 && isEnabledtransform(transformName)) {
            Transform::applyTransform(transformId, ir.get().getMainGraph());
          } else {
            logging::transform::err(
                "Transform '{}' does not exist or is not allowed to be used.",
                transformName);
          }
        }
      }
    }
  }
}

} // namespace popart
