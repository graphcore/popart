// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_PATTERNS_PATTERN_HPP_
#define POPART_WILLOW_INCLUDE_POPART_PATTERNS_PATTERN_HPP_

#include <memory>
#include <string>
#include <vector>

namespace popart {
class Op;
class Tensor;
struct OperatorIdentifier;

// Definition: A tensor is "touched" by a Pattern if
// its state is different at any point in the execution of the
// computation graph, different measured between
// 1) with Pattern applied
// 2) without Pattern applied
//
// As an example of touching a tensor: if a tensor is removed,
// we say that is has been touched (as applying the Pattern
// has changed it).
// Before a Pattern is applied, we always check that it does
// not touch an anchor tensor. Historical note: we previously
// only checked for deletion but this is too weak. Consider for
// example: a tensor might get a new consumer added to it which
// modifies it in-place.

class Pattern {
public:
  Pattern()          = default;
  virtual ~Pattern() = default;

  const std::string &getPatternName() const;

protected:
  std::string getReplacementOpName(Op *op, const std::string name) const;

  void transferBaseProperties(const Op *from, Op *to) const;
};

class PreAliasPattern : public Pattern {
public:
  PreAliasPattern()          = default;
  virtual ~PreAliasPattern() = default;

  // If this Pattern were to be applied at op, which
  // Tensors in the subgraph centered (rooted) on op
  // would be touched?
  virtual std::vector<const Tensor *> touches(Op *op) const = 0;

protected:
  // New op(s) created in replacement of old op will
  // inherit name and attributes of op they replace
  std::unique_ptr<Op> makeReplacementOp(const OperatorIdentifier &,
                                        Op *oldOp,
                                        const std::string name = "") const;

public:
  // New op(s) created in replacement of old op will
  // inherit name and attributes of op they replace
  Op *makeReplacementOpInIr(const OperatorIdentifier &,
                            Op *oldOp,
                            const std::string name = "") const;
  // Does this Pattern match the
  // sub-graph centered (rooted) on op?
  virtual bool matches(Op *op) const = 0;

  // Apply this Pattern, modifying the sub-graph
  // centered (rooted) on op
  virtual bool apply(Op *op) const = 0;

  // if applied to op, would there
  // be any anchored tensors touched?
  bool touchesAnchored(Op *) const;

private:
  static int tensor_counter;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_PATTERNS_PATTERN_HPP_
