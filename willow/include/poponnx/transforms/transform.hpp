#ifndef GUARD_NEURALNET_TRANSFORM_HPP
#define GUARD_NEURALNET_TRANSFORM_HPP

#include <typeinfo>

namespace poponnx {

class Graph;

class Transform {
public:
  Transform() {}
  virtual ~Transform() {}

  virtual bool apply(Graph &graph) const = 0;

  virtual std::size_t getId() const = 0;

  virtual std::string getName() const = 0;

  // apply a transformation to the given Ir
  static void applyTransform(std::size_t transformId, Graph &);

  // add a transform to the list of transforms
  static bool registerTransform(Transform *transform);
};

} // namespace poponnx

#endif
