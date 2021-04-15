// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TRANSFORM_HPP
#define GUARD_NEURALNET_TRANSFORM_HPP

#include <string>
#include <type_traits>
#include <typeinfo>

#include <popart/poparttracepoint.hpp>

namespace popart {

class Ir;
class Graph;

class Transform {
public:
  Transform() {}
  virtual ~Transform() {}

  virtual std::size_t getId() const = 0;

  virtual std::string getName() const = 0;

  /**
   * Base class method restricted to a Graph reference parameter and a bool
   * return type. This method exists for backwards compatibility but apply
   * methods can assume any signature if the templated `applyTransform` function
   * is used.
   **/
  virtual bool apply(Graph &graph) const = 0;

  /**
   * Call Transform::apply(graph) on a registered instance of a transform.
   *
   * As a side effect this method will both create a PVTI checkpoint and a
   * Poprithms timer for performance monitoring reasons. It will also output
   * some meaningful logs.
   *
   * NOTE: This version exists for backwards compatibility. Consider using the
   * generic version of `applyTransform` instead.
   *
   * \param transformId The ID with which the transform was registered.
   * \param args A graph to be passed to the transform.
   **/
  static void applyTransform(std::size_t transformId, Graph &graph);

  /**
   * Calling `Transform::applyTransform<TransformType>(ir, args)` is
   * functionally equivalent to:
   * ```
   * TransformType t;
   * t.apply(args...);
   * ```
   * However, as a side effect this method will both create a PVTI checkpoint
   * and a Poprithms timer for performance monitoring reasons. It will also
   * output some meaningful logs.
   *
   * This method of calling transforms is not restrictive in terms of either
   * the return type or argument type that the apply function has. However,
   * this implementation does not use the transform registry and creates a new
   * transform object for each call -- meaning your transform needs to be
   * stateless.
   *
   * NOTE: Consider calling this function via `popart::Ir::applyTransform`.
   *
   * \param ir The popart::Ir object in which the transformation is made.
   * \param args A list of arguments to be passed to the transform.
   * \return The value returned by the transform.
   **/
  template <typename T, typename... Args>
  static auto applyTransform(Ir &ir, Args... args)
      // Return type of this function matches return type of apply function.
      -> decltype(T().apply(std::forward<Args>(args)...));

  // add a transform to the list of transforms
  static bool registerTransform(Transform *transform);

private:
  // Generic helper function to wrap an apply call, used for both versions of
  // `applyTransform`. Note that the Args template argument can be inferred:
  // `Transform::applyTransformHelper<Autodiff>(autodiff, ir, std::ref(ir));`
  template <typename T, typename... Args>
  static auto applyTransformHelper(T &transform, Ir &ir, Args... args)
      // Return type of this function matches return type of apply function.
      -> decltype(transform.apply(std::forward<Args>(args)...));

  // Start a poprithms stopwatch for performance monitoring. This code is
  // wrapped in a helper function to avoid exposing poprithm headers.
  virtual void startStopwatch(Ir &ir);
  // Start a poprithms stopwatch for performance monitoring. This code is
  // wrapped in a helper function to avoid exposing poprithm headers.
  virtual void stopStopwatch(Ir &ir);
};

} // namespace popart

#include <popart/transforms/transform_impl.hpp>

#endif
