#ifndef GUARD_NEURALNET_PATTERN_HPP
#define GUARD_NEURALNET_PATTERN_HPP

#include <map>
#include <popart/names.hpp>
#include <popart/opidentifier.hpp>
#include <popart/tensornames.hpp>

namespace popart {

// All patterns which are run before any tensor aliasing has been performed
enum class PreAliasPatternType {
  PREUNIREPL = 0,
  POSTNREPL,
  SOFTMAXGRADDIRECT,
  NLLLWITHSOFTMAXGRADDIRECT,
  SPLITCONVBIAS,
  OPTOIDENTITY,
  SUBTRACTARG1GRADOP,
  MULARGGRADOP,
  RECIPROCALGRADOP,
  DIVARG0GRADOP,
  DIVARG1GRADOP,
  SINGRADOP,
  COSGRADOP,
  TANTOSINOVERCOS,
  SQRTGRADOP,
  EXPGRADOP,
  LOGGRADOP,
  COSHOP,
  LOGSOFTMAXOP,
  GEMMDECOMPOSITION,
  NEGATIVEONESCALE,
  PADSUM,
  ABSGRADOP,
  SPLITGATHER,
  CONVDATAGRAD,
  SUMTOADD,
  SPLITGRADOPTOCONCAT,
  SPLITOP,
  POWARG0GRADOP,
  POWARG1GRADOP,
  CONTIGUATEIPUCOPYINDICES,
  MATMULOP,
  MATMULLHSGRADOP,
  MATMULRHSGRADOP,
  SGD1DECOMPOSE
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

  void initialise(std::string pattern_name);

protected:
  std::string getReplacementOpName(Op *op, const std::string name) const;

  void transferBaseProperties(Op *from, Op *to) const;

private:
  std::string pattern_name;
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
