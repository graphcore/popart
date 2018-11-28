#ifndef GUARD_NEURALNET_RECOMPUTE_HPP
#define GUARD_NEURALNET_RECOMPUTE_HPP

#include <poponnx/transforms/transform.hpp>

namespace willow {

class Recompute : public Transform {
public:
  Recompute() : Transform() {}
  virtual ~Recompute() override {}

  virtual bool apply(Ir &ir) override;
};

} // namespace willow

#endif
