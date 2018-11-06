#ifndef GUARD_NEURALNET_PATTERNS_HPP
#define GUARD_NEURALNET_PATTERNS_HPP

#include <map>
#include <willow/names.hpp>

namespace willow {

enum class PatternType { PREUNIREPL = 0, POSTNREPL, SOFTMAXGRADDIRECT };

class PatternTypes {
public:
  PatternTypes();
  const PatternType &get(std::string op_type) const;
  const std::string &get(PatternType opType) const;

private:
  std::map<std::string, PatternType> opTypes_;
  std::map<PatternType, std::string> strings_;
};

PatternTypes initPatternTypes();
const PatternTypes &getPatternTypes();

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
// only checked for deletion but this seems too weak. Consider for
// example: a tensor might get a new consumer added to it which
// modifies it in-place.

class Pattern {
public:
  Pattern()          = default;
  virtual ~Pattern() = default;

  // Does this Pattern match the sub-graph centered (rooted)
  // on op?
  virtual bool matches(const Op *op) const = 0;
  // If this Pattern were to be applied at op, which
  // Tensors in the subgraph centered (rooted) on op
  // would be touched?
  virtual std::vector<const Tensor *> touches(const Op *op) const = 0;
  // Apply this Pattern, modifying the sub-graph centered (rooted)
  // on op
  virtual void apply(Op *op) const = 0;
  // if applied to op, would there
  // be any anchored tensors touched?
  bool touchesAnchored(const Op *) const;
};

// {(a), (b), (c)} -> [op0] -> (d) -> [op1] -> {(e), (f)}
//                    ====================>
// {(a), (b), (c)} ->        [op01]         -> {(e), (f)}
class FuserPattern : public Pattern {
public:
  virtual bool matches(const Op *) const override final;
  // Only (d) is touched. Therefore, a Pattern where [op1] and
  // [op01] perform inplace changes to an input tensor should
  // not inherit from FuserPattern.
  virtual std::vector<const Tensor *> touches(const Op *) const override final;
  virtual void apply(Op *) const override final;

private:
  // OpType of op0 in schematic
  virtual OpType get0() const = 0;
  // OpType of op1 in schematic
  virtual OpType get1() const = 0;
  // how to create a new op01 and move it into Ir
  virtual OpId moveMergedIntoIr(Op *baseOp) const = 0;
};

// consider,
// (label), (probs) -> [NLLGrad]
// [NllGrad] -> (d_probs)
// (d_probs), (probs) -> [SoftmaxGrad] -> (d_acts).
// This pattern replaces this with,
// (label), (probs) -> [SoftmaxGradDirect] -> (d_acts).
class SoftmaxGradDirect : public FuserPattern {
private:
  virtual OpType get0() const override final;
  virtual OpType get1() const override final;
  virtual OpId moveMergedIntoIr(Op *baseOp) const override final;
};

// remove ()->[] where () is a Tensor and [] is an Op and ()->[]
// forms part of [.]->()->[]->(.). after this, this section will
// be [.]->(.). [] is the root op of the pattern.
// [] IS THE IDENTITY OP
// () must be consumed by only []. This constraint can be
//    removed, shouldn't be too hard, although care needed
//    with topo cons (as always)
class PreUniRepl : public Pattern {
public:
  PreUniRepl()                   = default;
  virtual ~PreUniRepl() override = default;

  // Pad with pad size zero
  // Sum with one input
  virtual bool matches(const Op *) const override final;
  //  Only tensor (), which is deleted, is touched
  virtual std::vector<const Tensor *> touches(const Op *) const override final;
  virtual void apply(Op *) const override final;
};

// consider,
// (ori) -> [*] -> {(rep1), (rep2), (rep3)}
// where rep1 = ori, rep2 = ori, rep3 = ori
// We call the Op [*] an N-replicator.
// It is similar to the identity in PreUniRepl, but it
// has N=3 copies of the input instead of N=1.
// if (ori) -> {[op0], [*], [op1]}, and
// (rep1) -> {[op0], [op2], [op2]}, and
// (rep2) -> {[op2], [op3]}
// (rep3) -> {}
// then this should be replaced by
// (ori) -> {[op0], [op0], [o1], [op2], [op2], [op2], [o3]}
// removals : [*], (rep1), (rep2), (rep3)
// [*] is the root of the pattern
// there are checks that the consumer dependecices of
// ori, rep1, rep2 and pre3 can be merged
class PostNRepl : public Pattern {
public:
  PostNRepl()                   = default;
  virtual ~PostNRepl() override = default;

  // AddGrad (where N = 2)
  // Pad with pad size zero (where N = 1) *€
  // Sum with one input (where N = 1) *€
  virtual bool matches(const Op *) const override final;
  // rep1, rep2 and rep3 are touched (as they are deleted)
  // ori might be touched, if one its new consumers performs
  // an inplace modification to it.
  virtual std::vector<const Tensor *> touches(const Op *) const override final;
  virtual void apply(Op *) const override final;

private:
  // of all the tensors to be merged into 1 (ori, rep1, rep2, re3)
  // 1) how many of them have a "last" consumer,
  // 2) how many of them have weak topological constraints,
  // 3) if there is a single one with a "last" consumer, who is it?
  class TopoBundle {
  public:
    int nTopoLasts{0};
    int nWeakTopoCons{0};
    Op *lastCon{nullptr};
  };
  TopoBundle getTopoConInfo(const Op *op) const;

  // *€ : this pattern matches and removes []->()
  // whereas PreUniRepl matches and removes ()->[].
  // This pattern can be considered PostUniRepl when N = 1
};

} // namespace willow

#endif
