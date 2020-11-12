// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_BINARY_GRAD_OP_PATTERN_HPP
#define GUARD_NEURALNET_BINARY_GRAD_OP_PATTERN_HPP

#include <popart/patterns/patterns.hpp>

namespace popart {

// Base op pattern for replacing a binary op gradient op with a series of ops
class BinaryGradOpPattern : public PreAliasPattern {
public:
  // If this Pattern were to be applied at op, which
  // Tensors in the subgraph centered (rooted) on op
  // would be touched?
  std::vector<const Tensor *> touches(Op *) const override;

  // Removes all inputs of the orginal op, calls makeReplacementOps with the
  // relevant tensors, creates the ReduceSumOp which takes as an input the
  // tensor returned by makeReplacementOps and deletes original op
  bool apply(Op *) const override final;

protected:
  // Implements the replacement ops using the original op and specified tensors
  // and returns the id of the final tensor (which is the input into the reduce
  // sum to allow for np-style broadcasting)
  // The implementation is *not* responsible for disconnecting or erasing the
  // original op, or the reduce sum for np-style broadcasting.
  virtual TensorId makeAllReplacementOps(Op *op,
                                         Tensor *grad_in,
                                         Tensor *fwd_in0,
                                         Tensor *fwd_in1,
                                         Tensor *fwd_out) const = 0;
};

} // namespace popart

#endif
