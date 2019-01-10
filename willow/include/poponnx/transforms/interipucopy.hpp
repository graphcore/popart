#ifndef GUARD_NEURALNET_INTERIPUCOPY_HPP
#define GUARD_NEURALNET_INTERIPUCOPY_HPP

#include <poponnx/transforms/transform.hpp>

namespace poponnx {

using IpuNumber = int64_t;

IpuNumber getIpuNumber(const Op *op);

class InterIpuCopy : public Transform {
public:
  static std::size_t id();

  InterIpuCopy() : Transform() {}
  virtual ~InterIpuCopy() override {}

  virtual bool apply(Ir &ir) const override final;

  virtual std::size_t getId() const override final { return id(); }

  virtual std::string getName() const override final { return "InterIpuCopy"; }
};

} // namespace poponnx

#endif
