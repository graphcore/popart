// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_HOSTIOSETUP_HPP
#define GUARD_NEURALNET_HOSTIOSETUP_HPP

#include <popart/op.hpp>
#include <popart/transforms/transform.hpp>

namespace popart {

class HostIOSetup : public Transform {
public:
  static std::size_t id(int);

  HostIOSetup(int pass_) : Transform(), pass(pass_) {}
  virtual ~HostIOSetup() override {}

  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(pass); }

  virtual std::string getName() const final {
    return "HostIOSetup " + std::to_string(pass);
  }

private:
  void setupHostLoadOps(Tensor *) const;
  void setupHostStoreOps(Tensor *) const;

private:
  int pass;
};

} // namespace popart

#endif
