#ifndef GUARD_NEURALNET_SHARDING_CHECK_HPP
#define GUARD_NEURALNET_SHARDING_CHECK_HPP

#include <poponnx/transforms/transform.hpp>

namespace poponnx {

class VirtualGraphCheck : public Transform {
public:
  static std::size_t id();

  VirtualGraphCheck() : Transform() {}
  virtual ~VirtualGraphCheck() override {}

  virtual bool apply(Graph &graph) const override final;

  virtual std::size_t getId() const override final { return id(); }

  virtual std::string getName() const override final {
    return "VirtualGraphCheck";
  }
};

} // namespace poponnx

#endif
