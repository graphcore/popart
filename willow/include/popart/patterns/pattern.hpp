// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_PATTERN_HPP
#define GUARD_NEURALNET_PATTERN_HPP

#include <map>
#include <popart/names.hpp>
#include <popart/opidentifier.hpp>
#include <popart/tensornames.hpp>

namespace popart {

// All patterns which are run before any tensor aliasing has been performed
enum class PreAliasPatternType {
  PreUniRepl = 0,
  PostNRepl,
  SoftmaxGradDirect,
  NLLLWithSoftmaxGradDirect,
  SplitConvBias,
  OptoIdentity,
  SubtractArg1GradOp,
  MulArgGradOp,
  ReciprocalGradOp,
  DivArg0GradOp,
  DivArg1GradOp,
  SinGradOp,
  CosGradOp,
  TanToSinOverCos,
  SqrtGradOp,
  ExpGradOp,
  Expm1GradOp,
  Log1pGradOp,
  LogGradOp,
  CoshOp,
  GemmDecomposition,
  NegativeOneScale,
  PadSum,
  AbsGradOp,
  SplitGather,
  ConvDataGrad,
  SumtoAdd,
  SplitGradOpToConcat,
  SplitOp,
  PowArg0GradOp,
  PowArg1GradOp,
  ContiguateIPUCopyIndices,
  MatMulOp,
  MatMulLHSGradOp,
  MatMulRHSGradOp,
  SGD1Decompose,
  LSTMOp,
  InitAccumulate,
  UpsampleToResize,
  AcosOpPattern,
  AcoshOpPattern,
  RandomNormalLikeOpPattern,
  RandomUniformLikeOpPattern,
  ZerosLikeOpPattern,
  ConvTranspose,
  AsinhOpPattern,
  AtanhOpPattern
};

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

  void transferBaseProperties(Op *from, Op *to) const;
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

#endif
