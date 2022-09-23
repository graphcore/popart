// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_HOSTIOSETUP_HPP_
#define POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_HOSTIOSETUP_HPP_

#include <cstddef>
#include <string>
#include <popart/transforms/transform.hpp>

namespace popart {
class AliasModel;
class Graph;
class Tensor;

class HostIOSetup : public Transform {
public:
  static std::size_t id(int);

  HostIOSetup(int pass_) : Transform(), pass(pass_) {}
  ~HostIOSetup() override {}

  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(pass); }

  virtual std::string getName() const final {
    return "HostIOSetup " + std::to_string(pass);
  }

private:
  void setupHostLoadOps(Tensor *, AliasModel &) const;
  void setupHostStoreOps(Tensor *, AliasModel &) const;

private:
  int pass;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_HOSTIOSETUP_HPP_
