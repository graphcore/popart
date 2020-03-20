// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_HOSTREDUCE_HPP
#define GUARD_NEURALNET_HOSTREDUCE_HPP

#include <popart/transforms/transform.hpp>

namespace popart {

class HostReduce : public Transform {
private:
  Op *insertGradCopyToHostOp(Op *varUpdateOp, Graph &graph, int counter) const;
  Op *
  insertGradCopyFromHostOp(Op *varUpdateOp, Graph &graph, int counter) const;
  Op *insertVarCopyOp(Op *varUpdateOp, Graph &graph, int counter) const;

  void verifySessionOptions(const SessionOptions &) const;

public:
  static std::size_t id();

  HostReduce() : Transform() {}

  virtual bool apply(Graph &) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "HostReduce"; }
};

} // namespace popart

#endif
