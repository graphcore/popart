// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TRANSFORM_IMPL_HPP
#define GUARD_NEURALNET_TRANSFORM_IMPL_HPP

#include <popart/ir.hpp>
#include <popart/transforms/transform.hpp>

namespace popart {

// Implementation of Transform::applyTransform.
template <typename T, typename... Args>
auto Transform::applyTransform(Ir &ir, Args... args)
    -> decltype(T().apply(std::forward<Args>(args)...)) {
  T transform;
  return Transform::applyTransformHelper<T, Args...>(transform, ir, args...);
}

// Implementation of Transform::applyTransformHelper.
template <typename T, typename... Args>
auto Transform::applyTransformHelper(T &transform, Ir &ir, Args... args)
    -> decltype(transform.apply(std::forward<Args>(args)...)) {

  static_assert(std::is_base_of<Transform, T>(),
                "[Transform::applyTransformHelper] Template argument T must be "
                "a class derived from Transform");

  transform.startStopwatch(ir);

  PopartTracepoint tp(
      logging::format("Applying transform '{}'", transform.getName()));
  logging::transform::info("Applying Graph transform {}", transform.getName());

  auto result = transform.apply(std::forward<Args>(args)...);

  transform.stopStopwatch(ir);

  return result;
}

} // namespace popart

#endif