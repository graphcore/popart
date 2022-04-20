// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SERIALIZE_MATMULS_HPP
#define GUARD_NEURALNET_SERIALIZE_MATMULS_HPP

#include <cstddef>
#include <string>
#include <popart/transforms/transform.hpp>

namespace popart {
class Graph;

class SerializeMatMuls : public Transform {
public:
  static std::size_t id();

  SerializeMatMuls() : Transform() {}
  virtual ~SerializeMatMuls() override {}
  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "SerializeMatMuls"; }

private:
};

} // namespace popart

#endif // GUARD_NEURALNET_SERIALIZE_MATMULS_HPP
