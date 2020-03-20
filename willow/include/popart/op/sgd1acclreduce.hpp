// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SGD1ACCLREDUCEOP_HPP
#define GUARD_NEURALNET_SGD1ACCLREDUCEOP_HPP

#include <popart/op/varupdate.hpp>

namespace popart {

class SGD1AcclReduceOp : public VarUpdateWithoutUpdaterOp {

public:
  SGD1AcclReduceOp(const TensorId &acclToReduce, const Op::Settings &);

  std::unique_ptr<Op> clone() const final;
  std::unique_ptr<Op> cloneWithNewName(const TensorId &newName) const final;
  std::map<InIndex, TensorId> optimizerInputs() const final;
  void appendAttributes(OpSerialiserBase &) const final;

  // TODO https://phabricator.sourcevertex.net/T12562 for outlining this
  bool isOutlineable() const final { return false; }
  float getSubgraphValue() const final { return getHighSubgraphValue(); }
};

} // namespace popart

#endif
