#ifndef GUARD_NEURALNET_PATTERN_HPP
#define GUARD_NEURALNET_PATTERN_HPP

#include <map>
#include <poponnx/names.hpp>

namespace poponnx {

enum class PatternType {
  // Before the inplace Patterns:
  PREUNIREPL = 0,
  POSTNREPL,
  SOFTMAXGRADDIRECT,
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
  LOGSOFTMAXOP,
  COSHOP,
  GEMMDECOMPOSITION,
  // The patterns which can handle topological constraints:
  INPLACE0,
  INPLACEALL
};

enum class PatternPhase {
  PRETOPOCONS = 0,
  // To create a Pattern which correctly handles
  // topological constraints requires caution!
  WITHTOPOCONS
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
  // Does this Pattern match the
  // sub-graph centered (rooted) on op?
  virtual bool matches(Op *op) const = 0;
  // If this Pattern were to be applied at op, which
  // Tensors in the subgraph centered (rooted) on op
  // would be touched?
  virtual std::vector<const Tensor *> touches(Op *op) const = 0;
  // Apply this Pattern, modifying the sub-graph
  // centered (rooted) on op
  virtual bool apply(Op *op) const = 0;
  // if applied to op, would there
  // be any anchored tensors touched?
  bool touchesAnchored(Op *) const;
  // What phase will this Pattern be run at?
  virtual PatternPhase phase() const = 0;

  static TensorId createIntermediateTensorId(TensorId base_id);

  void initialise(std::string pattern_name);

  const std::string &getPatternName() const;

private:
  static int tensor_counter;
  std::string pattern_name;
};

} // namespace poponnx

#endif
