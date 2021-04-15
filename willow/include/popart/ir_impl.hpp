// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_IR_IMPL_HPP
#define GUARD_NEURALNET_IR_IMPL_HPP

#include <popart/ir.hpp>
#include <popart/transforms/transform.hpp>

namespace popart {

// Implementation of Ir::applyTransform.
template <typename T, typename... Args>
auto Ir::applyTransform(Args... args)
    -> decltype(T().apply(std::forward<Args>(args)...)) {
  return Transform::applyTransform<T, Args...>(*this,
                                               std::forward<Args>(args)...);
}

// Implementation of Ir::applyTransformIfEnabled.
template <typename T, typename... Args>
auto Ir::applyTransformIfEnabled(Args... args)
    -> nonstd::optional<decltype(T().apply(std::forward<Args>(args)...))> {
  if (isEnabledTransform(T::id())) {
    return applyTransform<T, Args...>(std::forward<Args>(args)...);
  } else {
    return nonstd::optional<decltype(T().apply(std::forward<Args>(args)...))>();
  }
}

} // namespace popart

namespace std {
template <> struct hash<popart::Ir> {
  std::size_t operator()(const popart::Ir &ir) const;
};

template <> struct hash<popart::IrBundle> {
  std::size_t operator()(const popart::IrBundle &irBundle) const;
};

} // namespace std

#endif
