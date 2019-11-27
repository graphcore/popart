#ifndef GUARD_NEURALNET_PINGPONG_HPP
#define GUARD_NEURALNET_PINGPONG_HPP

#include <popart/op.hpp>
#include <popart/transforms/transform.hpp>

namespace popart {

using IpuNumber = int64_t;

IpuNumber getIpuNumber(const Op *op);

class PingPong : public Transform {
public:
  static std::size_t id(int);

  PingPong(int pass_) : Transform(), pass(pass_) {}
  virtual ~PingPong() override {}

  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(pass); }

  TensorId generateAllocTensorId(Tensor *tensor) const;

  TensorId generateLoadedTensorId(Tensor *tensor, int64_t load_index) const;

  TensorId generateCacheArgTensorId(TensorId tid, VGraphId vgid) const;

  virtual std::string getName() const final {
    return "PingPong " + std::to_string(pass);
  }

  float costFn(Op *op) const;

private:
  int pass;
};

} // namespace popart

#endif
