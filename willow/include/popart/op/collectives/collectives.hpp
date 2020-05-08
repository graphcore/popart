// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_COLLECTIVES_HPP
#define GUARD_NEURALNET_COLLECTIVES_HPP

#include <popart/op.hpp>

namespace popart {

class CollectivesBaseOp : public Op {
public:
  CollectivesBaseOp(const OperatorIdentifier &opid,
                    const Op::Settings &settings);

  // Input to gather/reduce/scatter
  static InIndex getInIndex() { return 0; }

  // Tensor to backtrack collective ops that have to coordinate with each other
  static InIndex getCollectiveLinkedIndex() { return 1; }

  // Gathered/reduced/scattered output
  static OutIndex getOutIndex() { return 0; }
};

} // namespace popart

#endif
