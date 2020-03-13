// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CACHESETUP_HPP
#define GUARD_NEURALNET_CACHESETUP_HPP

#include <popart/op.hpp>
#include <popart/transforms/transform.hpp>

namespace popart {

class CacheSetup : public Transform {
public:
  static std::size_t id();

  CacheSetup() : Transform() {}
  virtual ~CacheSetup() override {}

  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "CacheSetup"; }
};

} // namespace popart

#endif
