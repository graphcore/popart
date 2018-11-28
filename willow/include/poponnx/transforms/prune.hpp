#ifndef GUARD_NEURALNET_PRUNE_HPP
#define GUARD_NEURALNET_PRUNE_HPP

#include <poponnx/transforms/transform.hpp>

namespace willow {

class Prune : public Transform {
public:
  Prune() : Transform() {}
  virtual ~Prune() override {}

  virtual bool apply(Ir &ir) override;
};

} // namespace willow

#endif
