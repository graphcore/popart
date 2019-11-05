#ifndef GUARD_NEURALNET_HOSTREDUCE_HPP
#define GUARD_NEURALNET_HOSTREDUCE_HPP

#include <popart/transforms/transform.hpp>

namespace popart {

class HostReduce : public Transform {
public:
  static std::size_t id();

  HostReduce() : Transform() {}
  virtual ~HostReduce() = default;

  virtual bool apply(Graph &) const final;

  virtual std::size_t getId() const override final { return id(); }

  virtual std::string getName() const override final { return "HostReduce"; }
};

} // namespace popart

#endif
