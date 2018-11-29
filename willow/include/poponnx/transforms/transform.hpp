#ifndef GUARD_NEURALNET_TRANSFORM_HPP
#define GUARD_NEURALNET_TRANSFORM_HPP

namespace poponnx {

class Ir;

class Transform {
public:
  Transform() {}
  virtual ~Transform() {}

  virtual bool apply(Ir &ir) = 0;
};

} // namespace poponnx

#endif
