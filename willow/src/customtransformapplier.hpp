// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_SRC_CUSTOMTRANSFORMAPPLIER_HPP_
#define POPART_WILLOW_SRC_CUSTOMTRANSFORMAPPLIER_HPP_

#include <cstddef>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace popart {

class Graph;
class Ir;
/**
 * The CustomTransformApplier is used to help determine which user-defined
 * transforms are inserted at specific points in the transform sequence by
 * reading session options.
 *
 * The transforms list can be executed by configuring the
 * customTransformApplierSettings of session options.
 *
 * Only existing or user-defined transforms can be configured and executed.
 */
class CustomTransformApplier {
public:
  CustomTransformApplier(Ir &ir) : ir(ir) {}
  ~CustomTransformApplier() {}

  /**
   * Add a check point to apply custom transforms
   *
   * \param checkPoint The name of the check point.
   */
  void applyCustomTransforms(const std::string &checkPoint);

private:
  // Predefined blacklist of tranforms that are not allowed to be used by users
  bool isEnabledtransform(const std::string &name);

  std::reference_wrapper<Ir> ir;
};

} // namespace popart

#endif // POPART_WILLOW_SRC_CUSTOMTRANSFORMAPPLIER_HPP_
