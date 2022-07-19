// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_PATTERNS_SEQUENCEEXPANDER_HPP_
#define POPART_WILLOW_INCLUDE_POPART_PATTERNS_SEQUENCEEXPANDER_HPP_

#include <memory>
#include <vector>
#include <popart/patterns/pattern.hpp>

namespace popart {
class Op;
class Tensor;

class SequenceExpander : public PreAliasPattern {
public:
  // If this Pattern were to be applied at op, which
  // Tensors in the subgraph centered (rooted) on op
  // would be touched?
  std::vector<const Tensor *> touches(Op *op) const final;
  // Apply this Pattern, modifying the sub-graph
  // centered (rooted) on op
  bool apply(Op *op) const final;

private:
  // Replace the given op with the returned sequence of ops
  //
  // The returned ops' input and output will be connected on tensor index 0.
  //
  // When removing n-ary ops, we assume that the input tensors should be mapped
  // to the output tensors by the tensor index.
  virtual std::vector<std::unique_ptr<Op>> sequence(Op *op) const = 0;

  // `seq` contains at least one op
  bool expand(std::vector<std::unique_ptr<Op>> &seq, Op *op) const;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_PATTERNS_SEQUENCEEXPANDER_HPP_
