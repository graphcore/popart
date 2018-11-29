#ifndef GUARD_NEURALNET_RECOMPUTE_HPP
#define GUARD_NEURALNET_RECOMPUTE_HPP

#include <poponnx/transforms/transform.hpp>

namespace poponnx {

class Recompute : public Transform {
public:
  static std::size_t id();

  Recompute() : Transform() {}
  virtual ~Recompute() override {}

  virtual bool apply(Ir &ir) const override final;

  virtual std::size_t getId() const override final { return id(); }

  virtual std::string getName() const override final { return "Recompute"; }
};

} // namespace poponnx

#endif
